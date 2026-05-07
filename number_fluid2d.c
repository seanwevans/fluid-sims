// number_fluid_2d.c
//
// Realtime multi-threaded layout + non-blocking raylib visualization.
//
// Windows / raylib w64devkit:
//   gcc -o number_graph_raylib_realtime_mt_colors.exe number_graph_raylib_realtime_mt_colors.c \
//     C:\raylib\raylib\src\raylib.rc.data -s -static -O3 -std=c99 \
//     -Wall -Wshadow -Wunused-parameter \
//     -IC:\raylib\raylib\src -Iexternal -DPLATFORM_DESKTOP -L. \
//     -lraylib -lopengl32 -lgdi32 -lwinmm -lshcore -lpthread
//
// Linux:
//   gcc -Ofast -march=native -flto -std=c99 number_graph_raylib_realtime_mt_colors.c \
//     -lraylib -lm -ldl -lpthread -lGL -lrt -lX11 -o number_graph_raylib_realtime_mt_colors
//
// Run:
//   ./number_graph_raylib_realtime_mt_colors [max_number] [threads]
//
// Examples:
//   ./number_graph_raylib_realtime_mt_colors 65536 16
//   ./number_graph_raylib_realtime_mt_colors 131072 24
//
// Controls:
//   right-drag pan | wheel zoom | space pause sim | e edges | r reset camera
//   +/- sim substeps/frame cap | c cycle color scheme

#include "raylib.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int count;
    int crossing;
    int trip;
} Barrier;

static void barrier_init(Barrier *b, int count) {
    pthread_mutex_init(&b->mutex, NULL);
    pthread_cond_init(&b->cond, NULL);
    b->count = count;
    b->crossing = 0;
    b->trip = 0;
}

static void barrier_wait(Barrier *b) {
    pthread_mutex_lock(&b->mutex);
    int trip = b->trip;

    b->crossing++;
    if (b->crossing == b->count) {
        b->crossing = 0;
        b->trip++;
        pthread_cond_broadcast(&b->cond);
    } else {
        while (trip == b->trip) pthread_cond_wait(&b->cond, &b->mutex);
    }

    pthread_mutex_unlock(&b->mutex);
}

static void barrier_destroy(Barrier *b) {
    pthread_cond_destroy(&b->cond);
    pthread_mutex_destroy(&b->mutex);
}

typedef struct {
    int from;
    int to;
} Edge;

typedef struct {
    float x, y;
    float vx, vy;
    float fx, fy;
} Body;

typedef struct {
    float cx, cy, half;
    float mass, mx, my;
    int body;
    int child[4];
} Quad;

typedef struct {
    Quad *q;
    int len;
    int cap;
    int overflow;
} QuadTree;

typedef struct Sim Sim;

typedef struct {
    int id;
    Sim *sim;
    float *local_fx;
    float *local_fy;
    int *stack;
} Worker;

struct Sim {
    Body *bodies;
    int body_count;

    Edge *edges;
    int edge_count;

    Quad *quad_storage;
    int quad_storage_cap;
    QuadTree tree;

    float *draw_x[2];
    float *draw_y[2];
    int draw_index;
    pthread_mutex_t draw_mutex;

    int thread_count;
    Worker *workers;
    pthread_t *threads;
    Barrier barrier;

    volatile int running;
    volatile int paused;
    volatile int reset_requested;
    volatile int substeps_per_publish;

    long long steps;
    int tree_overflow;
};

enum {
    COLOR_SCHEME_MINT = 0,
    COLOR_SCHEME_INDEX_BANDS,
    COLOR_SCHEME_LOG_BUCKETS,
    COLOR_SCHEME_RADIUS_BANDS,
    COLOR_SCHEME_XY_XOR,
    COLOR_SCHEME_COUNT
};

static const char *kColorSchemeNames[COLOR_SCHEME_COUNT] = {
    "mint",
    "index bands",
    "log buckets",
    "radius bands",
    "xy xor"
};

static const Color kPalette16[16] = {
    {123, 236, 178, 230},
    {102, 216, 238, 230},
    {167, 139, 250, 230},
    {244, 114, 182, 230},
    {248, 113, 113, 230},
    {251, 146, 60, 230},
    {250, 204, 21, 230},
    {163, 230, 53, 230},
    { 74, 222, 128, 230},
    { 45, 212, 191, 230},
    { 34, 211, 238, 230},
    { 96, 165, 250, 230},
    {129, 140, 248, 230},
    {192, 132, 252, 230},
    {244, 114, 182, 230},
    {251, 191, 36, 230}
};

static void *xmalloc(size_t n) {
    void *p = malloc(n);
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t n, size_t s) {
    void *p = calloc(n, s);
    if (!p) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
    return p;
}

static int detect_core_count(void) {
#if defined(_WIN32)
    return 8;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 1;
#endif
}

static Edge *generate_edges(int max_number, int *edge_count_out) {
    unsigned char *prime = xmalloc((size_t)max_number + 1);
    memset(prime, 1, (size_t)max_number + 1);
    prime[0] = prime[1] = 0;

    for (int p = 2; (int64_t)p * p <= max_number; p++) {
        if (!prime[p]) continue;
        for (int q = p * p; q <= max_number; q += p) prime[q] = 0;
    }

    int prime_edges = 0;
    for (int n = 2; n <= max_number; n++) prime_edges += prime[n] != 0;

    int divisor_edges = 0;
    for (int from = 2; from <= max_number; from++) divisor_edges += max_number / from - 1;

    int len = 0;
    int cap = prime_edges + divisor_edges;
    Edge *edges = xmalloc((size_t)cap * sizeof(*edges));

    for (int n = 2; n <= max_number; n++) {
        if (prime[n]) edges[len++] = (Edge){0, n - 1};
    }

    for (int from = 2; from <= max_number; from++) {
        for (int to = from + from; to <= max_number; to += from) {
            edges[len++] = (Edge){from - 1, to - 1};
        }
    }

    free(prime);
    *edge_count_out = len;
    return edges;
}

static int qt_new_node(QuadTree *t, float cx, float cy, float half) {
    if (t->len >= t->cap) {
        t->overflow = 1;
        return -1;
    }

    int id = t->len++;
    Quad *q = &t->q[id];

    q->cx = cx;
    q->cy = cy;
    q->half = half;
    q->mass = 0.0f;
    q->mx = 0.0f;
    q->my = 0.0f;
    q->body = -1;
    q->child[0] = q->child[1] = q->child[2] = q->child[3] = -1;

    return id;
}

static int qt_has_children(const Quad *q) {
    return q->child[0] >= 0;
}

static int qt_child_index_xy(const Quad *q, float x, float y) {
    int east = x >= q->cx;
    int south = y >= q->cy;
    return east | (south << 1);
}

static void qt_subdivide(QuadTree *t, int node) {
    Quad old = t->q[node];
    float h = old.half * 0.5f;

    int c0 = qt_new_node(t, old.cx - h, old.cy - h, h);
    int c1 = qt_new_node(t, old.cx + h, old.cy - h, h);
    int c2 = qt_new_node(t, old.cx - h, old.cy + h, h);
    int c3 = qt_new_node(t, old.cx + h, old.cy + h, h);

    Quad *q = &t->q[node];
    q->child[0] = c0;
    q->child[1] = c1;
    q->child[2] = c2;
    q->child[3] = c3;
}

static void qt_insert(QuadTree *t, int node, Body *bodies, int bi) {
    for (int depth = 0; depth < 44 && node >= 0 && !t->overflow; depth++) {
        Quad *q = &t->q[node];

        q->mass += 1.0f;
        q->mx += bodies[bi].x;
        q->my += bodies[bi].y;

        if (!qt_has_children(q)) {
            if (q->body < 0) {
                q->body = bi;
                return;
            }

            int old = q->body;
            q->body = -1;
            qt_subdivide(t, node);
            if (t->overflow) return;

            q = &t->q[node];

            int oi = qt_child_index_xy(q, bodies[old].x, bodies[old].y);
            qt_insert(t, q->child[oi], bodies, old);

            int ni = qt_child_index_xy(q, bodies[bi].x, bodies[bi].y);
            node = q->child[ni];
            continue;
        }

        int ci = qt_child_index_xy(q, bodies[bi].x, bodies[bi].y);
        node = q->child[ci];
    }
}

static QuadTree qt_build(Body *bodies, int n, Quad *storage, int storage_cap) {
    float minx = bodies[0].x, maxx = bodies[0].x;
    float miny = bodies[0].y, maxy = bodies[0].y;

    for (int i = 1; i < n; i++) {
        float x = bodies[i].x;
        float y = bodies[i].y;
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
    }

    float cx = 0.5f * (minx + maxx);
    float cy = 0.5f * (miny + maxy);
    float half = 0.5f * fmaxf(maxx - minx, maxy - miny) + 1.0f;

    QuadTree t;
    t.q = storage;
    t.len = 0;
    t.cap = storage_cap;
    t.overflow = 0;

    int root = qt_new_node(&t, cx, cy, half);
    if (root < 0) return t;

    for (int i = 0; i < n; i++) qt_insert(&t, root, bodies, i);

    return t;
}

static void init_bodies_circle(Body *bodies, int n, float radius) {
    bodies[0] = (Body){0};

    for (int i = 1; i < n; i++) {
        float a = (float)(2.0 * M_PI) * (float)(i - 1) / (float)(n - 1);
        bodies[i].x = cosf(a) * radius;
        bodies[i].y = sinf(a) * radius;
        bodies[i].vx = 0.0f;
        bodies[i].vy = 0.0f;
        bodies[i].fx = 0.0f;
        bodies[i].fy = 0.0f;
    }
}

static void publish_draw_buffers(Sim *sim) {
    int next;

    pthread_mutex_lock(&sim->draw_mutex);
    next = 1 - sim->draw_index;
    pthread_mutex_unlock(&sim->draw_mutex);

    for (int i = 0; i < sim->body_count; i++) {
        sim->draw_x[next][i] = sim->bodies[i].x;
        sim->draw_y[next][i] = sim->bodies[i].y;
    }

    pthread_mutex_lock(&sim->draw_mutex);
    sim->draw_index = next;
    pthread_mutex_unlock(&sim->draw_mutex);
}

static void apply_repulsion_from_tree(Worker *w, int bi) {
    Sim *sim = w->sim;
    const QuadTree *t = &sim->tree;
    Body *bodies = sim->bodies;
    int *stack = w->stack;

    const float theta2 = 0.75f * 0.75f;
    const float repulsion = 180.0f;
    const float softening = 4.0f;

    int sp = 0;
    stack[sp++] = 0;

    float bx = bodies[bi].x;
    float by = bodies[bi].y;
    float fx = 0.0f;
    float fy = 0.0f;

    while (sp > 0) {
        int node = stack[--sp];
        const Quad *q = &t->q[node];

        if (q->mass <= 0.0f) continue;
        if (!qt_has_children(q) && q->body == bi) continue;

        float inv_mass = 1.0f / q->mass;
        float comx = q->mx * inv_mass;
        float comy = q->my * inv_mass;

        float dx = bx - comx;
        float dy = by - comy;
        float d2 = dx * dx + dy * dy + softening;
        float width = q->half + q->half;

        if (!qt_has_children(q) || (width * width) < theta2 * d2) {
            float inv_d = 1.0f / sqrtf(d2);
            float f = repulsion * q->mass / d2;
            fx += dx * inv_d * f;
            fy += dy * inv_d * f;
            continue;
        }

        int c0 = q->child[0], c1 = q->child[1], c2 = q->child[2], c3 = q->child[3];
        if (c0 >= 0) stack[sp++] = c0;
        if (c1 >= 0) stack[sp++] = c1;
        if (c2 >= 0) stack[sp++] = c2;
        if (c3 >= 0) stack[sp++] = c3;
    }

    bodies[bi].fx += fx;
    bodies[bi].fy += fy;
}

static void worker_step(Worker *w) {
    Sim *sim = w->sim;

    const float link_length = 20.0f;
    const float spring_k = 0.0125f;
    const float softening = 4.0f;
    const float damping = 0.86f;
    const float dt = 0.50f;
    const float max_speed = 80.0f;
    const float max_speed2 = max_speed * max_speed;

    int id = w->id;
    int tc = sim->thread_count;

    int body_begin = 1 + (int)(((int64_t)(sim->body_count - 1) * id) / tc);
    int body_end = 1 + (int)(((int64_t)(sim->body_count - 1) * (id + 1)) / tc);

    int edge_begin = (int)(((int64_t)sim->edge_count * id) / tc);
    int edge_end = (int)(((int64_t)sim->edge_count * (id + 1)) / tc);

    for (int i = body_begin; i < body_end; i++) {
        sim->bodies[i].fx = 0.0f;
        sim->bodies[i].fy = 0.0f;
    }

    barrier_wait(&sim->barrier);

    if (id == 0) {
        sim->bodies[0].x = 0.0f;
        sim->bodies[0].y = 0.0f;
        sim->bodies[0].vx = 0.0f;
        sim->bodies[0].vy = 0.0f;
        sim->bodies[0].fx = 0.0f;
        sim->bodies[0].fy = 0.0f;

        sim->tree = qt_build(sim->bodies, sim->body_count, sim->quad_storage, sim->quad_storage_cap);
        sim->tree_overflow = sim->tree.overflow;
    }

    barrier_wait(&sim->barrier);

    if (!sim->tree_overflow) {
        for (int i = body_begin; i < body_end; i++) apply_repulsion_from_tree(w, i);
    }

    memset(w->local_fx, 0, (size_t)sim->body_count * sizeof(float));
    memset(w->local_fy, 0, (size_t)sim->body_count * sizeof(float));

    for (int e = edge_begin; e < edge_end; e++) {
        int src = sim->edges[e].from;
        int dst = sim->edges[e].to;

        float dx = sim->bodies[dst].x - sim->bodies[src].x;
        float dy = sim->bodies[dst].y - sim->bodies[src].y;
        float d2 = dx * dx + dy * dy + softening;
        float inv_d = 1.0f / sqrtf(d2);
        float d = d2 * inv_d;

        float f = spring_k * (d - link_length);
        float fx = dx * inv_d * f;
        float fy = dy * inv_d * f;

        if (src != 0) {
            w->local_fx[src] += fx;
            w->local_fy[src] += fy;
        }

        if (dst != 0) {
            w->local_fx[dst] -= fx;
            w->local_fy[dst] -= fy;
        }
    }

    barrier_wait(&sim->barrier);

    for (int i = body_begin; i < body_end; i++) {
        float fx = sim->bodies[i].fx;
        float fy = sim->bodies[i].fy;

        for (int t = 0; t < tc; t++) {
            Worker *other = &sim->workers[t];
            fx += other->local_fx[i];
            fy += other->local_fy[i];
        }

        float vx = (sim->bodies[i].vx + fx * dt) * damping;
        float vy = (sim->bodies[i].vy + fy * dt) * damping;

        float speed2 = vx * vx + vy * vy;
        if (speed2 > max_speed2) {
            float scale = max_speed / sqrtf(speed2);
            vx *= scale;
            vy *= scale;
        }

        sim->bodies[i].vx = vx;
        sim->bodies[i].vy = vy;
        sim->bodies[i].x += vx * dt;
        sim->bodies[i].y += vy * dt;
    }

    barrier_wait(&sim->barrier);

    if (id == 0) {
        sim->steps++;
        if ((sim->steps % sim->substeps_per_publish) == 0) publish_draw_buffers(sim);
    }

    barrier_wait(&sim->barrier);
}

static void *worker_main(void *userdata) {
    Worker *w = userdata;
    Sim *sim = w->sim;

    while (sim->running) {
        if (sim->paused) {
            if (w->id == 0) publish_draw_buffers(sim);
            for (volatile int spin = 0; spin < 500000; spin++) {}
            continue;
        }

        if (w->id == 0 && sim->reset_requested) {
            init_bodies_circle(sim->bodies, sim->body_count, sqrtf((float)sim->body_count) * 20.0f);
            sim->steps = 0;
            sim->reset_requested = 0;
            publish_draw_buffers(sim);
        }

        barrier_wait(&sim->barrier);
        worker_step(w);
    }

    return NULL;
}

static void sim_start(Sim *sim, int body_count, Edge *edges, int edge_count, int thread_count) {
    sim->body_count = body_count;
    sim->edge_count = edge_count;
    sim->edges = edges;
    sim->thread_count = thread_count;
    sim->running = 1;
    sim->paused = 0;
    sim->reset_requested = 0;
    sim->substeps_per_publish = 1;
    sim->steps = 0;
    sim->tree_overflow = 0;

    sim->bodies = xcalloc((size_t)body_count, sizeof(*sim->bodies));
    init_bodies_circle(sim->bodies, body_count, sqrtf((float)body_count) * 20.0f);

    sim->quad_storage_cap = body_count * 16 + 4096;
    sim->quad_storage = xmalloc((size_t)sim->quad_storage_cap * sizeof(*sim->quad_storage));

    sim->draw_x[0] = xmalloc((size_t)body_count * sizeof(float));
    sim->draw_x[1] = xmalloc((size_t)body_count * sizeof(float));
    sim->draw_y[0] = xmalloc((size_t)body_count * sizeof(float));
    sim->draw_y[1] = xmalloc((size_t)body_count * sizeof(float));
    sim->draw_index = 0;
    pthread_mutex_init(&sim->draw_mutex, NULL);

    for (int i = 0; i < body_count; i++) {
        sim->draw_x[0][i] = sim->bodies[i].x;
        sim->draw_y[0][i] = sim->bodies[i].y;
        sim->draw_x[1][i] = sim->bodies[i].x;
        sim->draw_y[1][i] = sim->bodies[i].y;
    }

    barrier_init(&sim->barrier, thread_count);

    sim->workers = xcalloc((size_t)thread_count, sizeof(*sim->workers));
    sim->threads = xmalloc((size_t)thread_count * sizeof(*sim->threads));

    for (int t = 0; t < thread_count; t++) {
        sim->workers[t].id = t;
        sim->workers[t].sim = sim;
        sim->workers[t].local_fx = xcalloc((size_t)body_count, sizeof(float));
        sim->workers[t].local_fy = xcalloc((size_t)body_count, sizeof(float));
        sim->workers[t].stack = xmalloc((size_t)sim->quad_storage_cap * sizeof(int));

        if (pthread_create(&sim->threads[t], NULL, worker_main, &sim->workers[t]) != 0) {
            fprintf(stderr, "pthread_create failed\n");
            exit(1);
        }
    }
}

static void sim_stop(Sim *sim) {
    sim->running = 0;

    for (int t = 0; t < sim->thread_count; t++) pthread_join(sim->threads[t], NULL);

    for (int t = 0; t < sim->thread_count; t++) {
        free(sim->workers[t].local_fx);
        free(sim->workers[t].local_fy);
        free(sim->workers[t].stack);
    }

    barrier_destroy(&sim->barrier);
    pthread_mutex_destroy(&sim->draw_mutex);

    free(sim->threads);
    free(sim->workers);
    free(sim->draw_x[0]);
    free(sim->draw_x[1]);
    free(sim->draw_y[0]);
    free(sim->draw_y[1]);
    free(sim->quad_storage);
    free(sim->bodies);
}

static int snapshot_draw_index(Sim *sim) {
    pthread_mutex_lock(&sim->draw_mutex);
    int idx = sim->draw_index;
    pthread_mutex_unlock(&sim->draw_mutex);
    return idx;
}

static void compute_bounds_xy(const float *x, const float *y, int n, float *minx, float *maxx, float *miny, float *maxy) {
    *minx = *maxx = x[0];
    *miny = *maxy = y[0];

    for (int i = 1; i < n; i++) {
        if (x[i] < *minx) *minx = x[i];
        if (x[i] > *maxx) *maxx = x[i];
        if (y[i] < *miny) *miny = y[i];
        if (y[i] > *maxy) *maxy = y[i];
    }
}

static Camera2D make_fit_camera_xy(const float *x, const float *y, int n) {
    float minx, maxx, miny, maxy;
    compute_bounds_xy(x, y, n, &minx, &maxx, &miny, &maxy);

    float bw = fmaxf(maxx - minx, 1.0f);
    float bh = fmaxf(maxy - miny, 1.0f);

    Camera2D camera = {0};
    camera.target = (Vector2){0.5f * (minx + maxx), 0.5f * (miny + maxy)};
    camera.offset = (Vector2){GetScreenWidth() * 0.5f, GetScreenHeight() * 0.5f};
    camera.rotation = 0.0f;

    float zx = (float)GetScreenWidth() / bw;
    float zy = (float)GetScreenHeight() / bh;
    camera.zoom = 0.88f * fminf(zx, zy);
    if (camera.zoom < 0.001f) camera.zoom = 0.001f;
    if (camera.zoom > 20.0f) camera.zoom = 20.0f;

    return camera;
}

static inline unsigned u32_log2_floor(unsigned x) {
    unsigned r = 0;
    while (x >>= 1u) r++;
    return r;
}

static inline Color point_color(int i, float xi, float yi, int scheme) {
    switch (scheme) {
        case COLOR_SCHEME_MINT:
            return (Color){123, 236, 178, 230};

        case COLOR_SCHEME_INDEX_BANDS:
            return kPalette16[(unsigned)i & 15u];

        case COLOR_SCHEME_LOG_BUCKETS: {
            unsigned bucket = u32_log2_floor((unsigned)(i + 1));
            return kPalette16[bucket & 15u];
        }

        case COLOR_SCHEME_RADIUS_BANDS: {
            float d2 = xi * xi + yi * yi;
            unsigned bucket = (unsigned)(d2 * 0.00006f);
            return kPalette16[bucket & 15u];
        }

        case COLOR_SCHEME_XY_XOR: {
            unsigned ax = (unsigned)((int)(fabsf(xi) * 0.035f));
            unsigned ay = (unsigned)((int)(fabsf(yi) * 0.035f));
            return kPalette16[(ax ^ ay) & 15u];
        }

        default:
            return (Color){123, 236, 178, 230};
    }
}

static void draw_points_fast_xy(const float *x, const float *y, int n, Camera2D camera, int color_scheme) {
    float zoom = camera.zoom;
    float ox = camera.offset.x;
    float oy = camera.offset.y;
    float tx = camera.target.x;
    float ty = camera.target.y;
    int w = GetScreenWidth();
    int h = GetScreenHeight();

    Color root = (Color){236, 178, 123, 255};

    if (zoom < 1.5f) {
        for (int i = 1; i < n; i++) {
            int sx = (int)((x[i] - tx) * zoom + ox);
            int sy = (int)((y[i] - ty) * zoom + oy);
            if ((unsigned)sx < (unsigned)w && (unsigned)sy < (unsigned)h) {
                DrawPixel(sx, sy, point_color(i, x[i], y[i], color_scheme));
            }
        }
    } else if (zoom < 5.0f) {
        for (int i = 1; i < n; i++) {
            int sx = (int)((x[i] - tx) * zoom + ox);
            int sy = (int)((y[i] - ty) * zoom + oy);
            if ((unsigned)sx < (unsigned)w && (unsigned)sy < (unsigned)h) {
                DrawRectangle(sx, sy, 2, 2, point_color(i, x[i], y[i], color_scheme));
            }
        }
    } else {
        BeginMode2D(camera);
        float r = fmaxf(1.0f / zoom, 0.35f);
        for (int i = 1; i < n; i++) {
            DrawCircleV((Vector2){x[i], y[i]}, r, point_color(i, x[i], y[i], color_scheme));
        }
        EndMode2D();
    }

    int rx = (int)((0.0f - tx) * zoom + ox);
    int ry = (int)((0.0f - ty) * zoom + oy);
    DrawCircle(rx, ry, 4.0f, root);
    
}

int main(int argc, char **argv) {
    int max_number = 1<<17;
    int thread_count = 0;

    if (argc > 1) max_number = atoi(argv[1]);
    if (argc > 2) thread_count = atoi(argv[2]);

    if (max_number < 2) max_number = 2;
    if (thread_count <= 0) thread_count = detect_core_count();
    if (thread_count < 1) thread_count = 1;
    if (thread_count > 256) thread_count = 256;
    if (thread_count > max_number - 1) thread_count = max_number - 1;

    int edge_count = 0;
    Edge *edges = generate_edges(max_number, &edge_count);

    fprintf(stderr, "nodes: %d\n", max_number);
    fprintf(stderr, "edges: %d\n", edge_count);
    fprintf(stderr, "threads: %d\n", thread_count);

    Sim sim;
    memset(&sim, 0, sizeof(sim));    
    sim_start(&sim, max_number, edges, edge_count, thread_count);

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(1600, 1000, "Prime-rooted multiples graph - realtime MT");
    SetTargetFPS(99999);

    int idx = snapshot_draw_index(&sim);
    Camera2D camera = make_fit_camera_xy(sim.draw_x[idx], sim.draw_y[idx], max_number);

    int draw_edges = 0;
    int color_scheme = COLOR_SCHEME_MINT;
    long long last_steps = 0;
    double sim_sps_smoothed = 0.0;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_SPACE)) sim.paused = !sim.paused;
        if (IsKeyPressed(KEY_E)) draw_edges = !draw_edges;
        if (IsKeyPressed(KEY_C)) color_scheme = (color_scheme + 1) % COLOR_SCHEME_COUNT;
        if (IsKeyPressed(KEY_R)) {
            idx = snapshot_draw_index(&sim);
            camera = make_fit_camera_xy(sim.draw_x[idx], sim.draw_y[idx], max_number);
        }

        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
            if (sim.substeps_per_publish < 64) sim.substeps_per_publish *= 2;
        }

        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
            if (sim.substeps_per_publish > 1) sim.substeps_per_publish /= 2;
        }

        if (IsKeyPressed(KEY_BACKSPACE)) sim.reset_requested = 1;

        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            Vector2 mouse_world_before = GetScreenToWorld2D(GetMousePosition(), camera);
            camera.zoom *= powf(1.12f, wheel);
            if (camera.zoom < 0.001f) camera.zoom = 0.001f;
            if (camera.zoom > 100.0f) camera.zoom = 100.0f;
            Vector2 mouse_world_after = GetScreenToWorld2D(GetMousePosition(), camera);
            camera.target.x += mouse_world_before.x - mouse_world_after.x;
            camera.target.y += mouse_world_before.y - mouse_world_after.y;
        }

        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Vector2 delta = GetMouseDelta();
            camera.target.x -= delta.x / camera.zoom;
            camera.target.y -= delta.y / camera.zoom;
        }

        camera.offset = (Vector2){GetScreenWidth() * 0.5f, GetScreenHeight() * 0.5f};

        idx = snapshot_draw_index(&sim);
        const float *x = sim.draw_x[idx];
        const float *y = sim.draw_y[idx];

        long long steps_now = sim.steps;
        double fps = (double)GetFPS();
        if (fps > 0.0) {
            double instant = (double)(steps_now - last_steps) * fps;
            sim_sps_smoothed = sim_sps_smoothed == 0.0 ? instant : sim_sps_smoothed * 0.90 + instant * 0.10;
        }
        last_steps = steps_now;

        BeginDrawing();
        ClearBackground((Color){8, 10, 14, 255});

        if (draw_edges) {
            BeginMode2D(camera);
            for (int e = 0; e < edge_count; e++) {
                int a = edges[e].from;
                int b = edges[e].to;
                DrawLineV((Vector2){x[a], y[a]}, (Vector2){x[b], y[b]}, (Color){70, 80, 100, 22});
            }
            EndMode2D();
        }

        draw_points_fast_xy(x, y, max_number, camera, color_scheme);

        DrawRectangle(12, 12, 620, 180, (Color){0, 0, 0, 155});
        DrawText(TextFormat("nodes: %d", max_number), 24, 24, 20, RAYWHITE);
        DrawText(TextFormat("edges: %d", edge_count), 24, 48, 20, RAYWHITE);
        DrawText(TextFormat("threads: %d", thread_count), 24, 72, 20, RAYWHITE);
        DrawText(TextFormat("sim steps: %lld", steps_now), 24, 96, 20, RAYWHITE);
        DrawText(TextFormat("sim steps/sec approx: %.1f", sim_sps_smoothed), 24, 120, 20, RAYWHITE);
        DrawText(TextFormat("render fps: %d | publish stride: %d | %s",
                            GetFPS(),
                            sim.substeps_per_publish,
                            sim.paused ? "paused" : "running"),
                 24, 144, 20, RAYWHITE);
        DrawText(TextFormat("color scheme: %s", kColorSchemeNames[color_scheme]),
                 24, 168, 20, RAYWHITE);

        DrawText("right-drag pan | wheel zoom | space pause | e edges | c colors | r refit | backspace reset | +/- publish stride",
                 24, GetScreenHeight() - 34, 18, GRAY);

        EndDrawing();
    }

    sim_stop(&sim);
    free(edges);

    CloseWindow();
    return 0;
}
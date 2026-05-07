// number_fluid_3d.c
//
// Differences vs the 2D version:
//   - Body, forces, velocity, integrator: x,y,z
//   - Quadtree -> Octree (8 children, 3D center-of-mass, 3D Barnes-Hut)
//   - Initial layout: Fibonacci sphere instead of unit circle
//   - Render: orbit Camera3D + cached view*projection matrix; points are
//     projected manually so the fast pixel/rect path still works
//   - Edges drawn as 2D screen-space lines after projection
//   - Right-drag orbits, middle-drag pans, wheel zooms in/out
//
// Windows / raylib w64devkit:
//   gcc -o number_fluid.exe number_fluid.c \
//     C:\raylib\raylib\src\raylib.rc.data -s -static -O3 -std=c99 \
//     -Wall -Wshadow -Wunused-parameter \
//     -IC:\raylib\raylib\src -Iexternal -DPLATFORM_DESKTOP -L. \
//     -lraylib -lopengl32 -lgdi32 -lwinmm -lshcore -lpthread
//
// Linux:
//   gcc -Ofast -march=native -flto -std=c99 number_fluid.c \
//     -lraylib -lm -ldl -lpthread -lGL -lrt -lX11 -o number_fluid
//
// Run:
//   ./number_fluid [max_number] [threads]
//
// Examples:
//   ./number_fluid 65536 16
//   ./number_fluid 131072 24
//
// Controls:
//   right-drag orbit | middle-drag pan | wheel zoom
//   space pause sim | e edges | c cycle color scheme | p chunky points
//   r refit camera  | backspace reset bodies | +/- publish stride

#include "raylib.h"
#include "raymath.h"

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
    float x, y, z;
    float vx, vy, vz;
    float fx, fy, fz;
} Body;

typedef struct {
    float cx, cy, cz, half;
    float mass, mx, my, mz;
    int body;
    int child[8];
} Oct;

typedef struct {
    Oct *q;
    int len;
    int cap;
    int overflow;
} OctTree;

typedef struct Sim Sim;

typedef struct {
    int id;
    Sim *sim;
    float *local_fx;
    float *local_fy;
    float *local_fz;
    int *stack;
} Worker;

struct Sim {
    Body *bodies;
    int body_count;

    Edge *edges;
    int edge_count;

    Oct *oct_storage;
    int oct_storage_cap;
    OctTree tree;

    float *draw_x[2];
    float *draw_y[2];
    float *draw_z[2];
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
    COLOR_SCHEME_XYZ_XOR,
    COLOR_SCHEME_COUNT
};

static const char *kColorSchemeNames[COLOR_SCHEME_COUNT] = {
    "mint",
    "index bands",
    "log buckets",
    "radius bands",
    "xyz xor"
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

static int ot_new_node(OctTree *t, float cx, float cy, float cz, float half) {
    if (t->len >= t->cap) {
        t->overflow = 1;
        return -1;
    }

    int id = t->len++;
    Oct *q = &t->q[id];

    q->cx = cx;
    q->cy = cy;
    q->cz = cz;
    q->half = half;
    q->mass = 0.0f;
    q->mx = 0.0f;
    q->my = 0.0f;
    q->mz = 0.0f;
    q->body = -1;
    for (int i = 0; i < 8; i++) q->child[i] = -1;

    return id;
}

static int ot_has_children(const Oct *q) {
    return q->child[0] >= 0;
}

static int ot_child_index_xyz(const Oct *q, float x, float y, float z) {
    int east  = x >= q->cx;
    int south = y >= q->cy;
    int back  = z >= q->cz;
    return east | (south << 1) | (back << 2);
}

static void ot_subdivide(OctTree *t, int node) {
    Oct old = t->q[node];
    float h = old.half * 0.5f;

    int c[8];
    int idx = 0;
    for (int kz = 0; kz < 2; kz++) {
        for (int ky = 0; ky < 2; ky++) {
            for (int kx = 0; kx < 2; kx++) {
                float ccx = old.cx + (kx ? h : -h);
                float ccy = old.cy + (ky ? h : -h);
                float ccz = old.cz + (kz ? h : -h);
                c[idx++] = ot_new_node(t, ccx, ccy, ccz, h);
            }
        }
    }

    Oct *q = &t->q[node];
    for (int i = 0; i < 8; i++) q->child[i] = c[i];
}

static void ot_insert(OctTree *t, int node, Body *bodies, int bi) {
    for (int depth = 0; depth < 44 && node >= 0 && !t->overflow; depth++) {
        Oct *q = &t->q[node];

        q->mass += 1.0f;
        q->mx += bodies[bi].x;
        q->my += bodies[bi].y;
        q->mz += bodies[bi].z;

        if (!ot_has_children(q)) {
            if (q->body < 0) {
                q->body = bi;
                return;
            }

            int old = q->body;
            q->body = -1;
            ot_subdivide(t, node);
            if (t->overflow) return;

            q = &t->q[node];

            int oi = ot_child_index_xyz(q, bodies[old].x, bodies[old].y, bodies[old].z);
            ot_insert(t, q->child[oi], bodies, old);

            int ni = ot_child_index_xyz(q, bodies[bi].x, bodies[bi].y, bodies[bi].z);
            node = q->child[ni];
            continue;
        }

        int ci = ot_child_index_xyz(q, bodies[bi].x, bodies[bi].y, bodies[bi].z);
        node = q->child[ci];
    }
}

static OctTree ot_build(Body *bodies, int n, Oct *storage, int storage_cap) {
    float minx = bodies[0].x, maxx = bodies[0].x;
    float miny = bodies[0].y, maxy = bodies[0].y;
    float minz = bodies[0].z, maxz = bodies[0].z;

    for (int i = 1; i < n; i++) {
        float x = bodies[i].x;
        float y = bodies[i].y;
        float z = bodies[i].z;
        if (x < minx) minx = x;
        if (x > maxx) maxx = x;
        if (y < miny) miny = y;
        if (y > maxy) maxy = y;
        if (z < minz) minz = z;
        if (z > maxz) maxz = z;
    }

    float cx = 0.5f * (minx + maxx);
    float cy = 0.5f * (miny + maxy);
    float cz = 0.5f * (minz + maxz);
    float spanx = maxx - minx;
    float spany = maxy - miny;
    float spanz = maxz - minz;
    float half = 0.5f * fmaxf(fmaxf(spanx, spany), spanz) + 1.0f;

    OctTree t;
    t.q = storage;
    t.len = 0;
    t.cap = storage_cap;
    t.overflow = 0;

    int root = ot_new_node(&t, cx, cy, cz, half);
    if (root < 0) return t;

    for (int i = 0; i < n; i++) ot_insert(&t, root, bodies, i);

    return t;
}

static void init_bodies_sphere(Body *bodies, int n, float radius) {
    bodies[0] = (Body){0};
    if (n <= 1) return;

    const float golden = (float)(M_PI * (3.0 - 2.2360679774997896964));  // pi*(3-sqrt(5))
    int m = n - 1;

    for (int i = 1; i < n; i++) {
        int k = i - 1;
        float t = (m == 1) ? 0.0f : (float)k / (float)(m - 1);
        float yy = 1.0f - 2.0f * t;
        float r = sqrtf(fmaxf(0.0f, 1.0f - yy * yy));
        float phi = golden * (float)k;

        bodies[i].x = cosf(phi) * r * radius;
        bodies[i].y = yy * radius;
        bodies[i].z = sinf(phi) * r * radius;
        bodies[i].vx = bodies[i].vy = bodies[i].vz = 0.0f;
        bodies[i].fx = bodies[i].fy = bodies[i].fz = 0.0f;
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
        sim->draw_z[next][i] = sim->bodies[i].z;
    }

    pthread_mutex_lock(&sim->draw_mutex);
    sim->draw_index = next;
    pthread_mutex_unlock(&sim->draw_mutex);
}

static void apply_repulsion_from_tree(Worker *w, int bi) {
    Sim *sim = w->sim;
    const OctTree *t = &sim->tree;
    Body *bodies = sim->bodies;
    int *stack = w->stack;

    const float theta2 = 0.75f * 0.75f;
    const float repulsion = 180.0f;
    const float softening = 4.0f;

    int sp = 0;
    stack[sp++] = 0;

    float bx = bodies[bi].x;
    float by = bodies[bi].y;
    float bz = bodies[bi].z;
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    while (sp > 0) {
        int node = stack[--sp];
        const Oct *q = &t->q[node];

        if (q->mass <= 0.0f) continue;
        if (!ot_has_children(q) && q->body == bi) continue;

        float inv_mass = 1.0f / q->mass;
        float comx = q->mx * inv_mass;
        float comy = q->my * inv_mass;
        float comz = q->mz * inv_mass;

        float dx = bx - comx;
        float dy = by - comy;
        float dz = bz - comz;
        float d2 = dx * dx + dy * dy + dz * dz + softening;
        float width = q->half + q->half;

        if (!ot_has_children(q) || (width * width) < theta2 * d2) {
            float inv_d = 1.0f / sqrtf(d2);
            float f = repulsion * q->mass / d2;
            fx += dx * inv_d * f;
            fy += dy * inv_d * f;
            fz += dz * inv_d * f;
            continue;
        }

        // Push all 8 children (skip empty slots defensively)
        for (int k = 0; k < 8; k++) {
            int c = q->child[k];
            if (c >= 0) stack[sp++] = c;
        }
    }

    bodies[bi].fx += fx;
    bodies[bi].fy += fy;
    bodies[bi].fz += fz;
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
    int body_end   = 1 + (int)(((int64_t)(sim->body_count - 1) * (id + 1)) / tc);
    int edge_begin = (int)(((int64_t)sim->edge_count * id) / tc);
    int edge_end   = (int)(((int64_t)sim->edge_count * (id + 1)) / tc);

    for (int i = body_begin; i < body_end; i++) {
        sim->bodies[i].fx = 0.0f;
        sim->bodies[i].fy = 0.0f;
        sim->bodies[i].fz = 0.0f;
    }

    barrier_wait(&sim->barrier);

    if (id == 0) {
        sim->bodies[0].x = 0.0f; sim->bodies[0].y = 0.0f; sim->bodies[0].z = 0.0f;
        sim->bodies[0].vx = 0.0f; sim->bodies[0].vy = 0.0f; sim->bodies[0].vz = 0.0f;
        sim->bodies[0].fx = 0.0f; sim->bodies[0].fy = 0.0f; sim->bodies[0].fz = 0.0f;

        sim->tree = ot_build(sim->bodies, sim->body_count, sim->oct_storage, sim->oct_storage_cap);
        sim->tree_overflow = sim->tree.overflow;
    }

    barrier_wait(&sim->barrier);

    if (!sim->tree_overflow) {
        for (int i = body_begin; i < body_end; i++) apply_repulsion_from_tree(w, i);
    }

    memset(w->local_fx, 0, (size_t)sim->body_count * sizeof(float));
    memset(w->local_fy, 0, (size_t)sim->body_count * sizeof(float));
    memset(w->local_fz, 0, (size_t)sim->body_count * sizeof(float));

    for (int e = edge_begin; e < edge_end; e++) {
        int src = sim->edges[e].from;
        int dst = sim->edges[e].to;

        float dx = sim->bodies[dst].x - sim->bodies[src].x;
        float dy = sim->bodies[dst].y - sim->bodies[src].y;
        float dz = sim->bodies[dst].z - sim->bodies[src].z;
        float d2 = dx * dx + dy * dy + dz * dz + softening;
        float inv_d = 1.0f / sqrtf(d2);
        float d = d2 * inv_d;

        float f = spring_k * (d - link_length);
        float fx = dx * inv_d * f;
        float fy = dy * inv_d * f;
        float fz = dz * inv_d * f;

        if (src != 0) {
            w->local_fx[src] += fx;
            w->local_fy[src] += fy;
            w->local_fz[src] += fz;
        }

        if (dst != 0) {
            w->local_fx[dst] -= fx;
            w->local_fy[dst] -= fy;
            w->local_fz[dst] -= fz;
        }
    }

    barrier_wait(&sim->barrier);

    for (int i = body_begin; i < body_end; i++) {
        float fx = sim->bodies[i].fx;
        float fy = sim->bodies[i].fy;
        float fz = sim->bodies[i].fz;

        for (int t = 0; t < tc; t++) {
            Worker *other = &sim->workers[t];
            fx += other->local_fx[i];
            fy += other->local_fy[i];
            fz += other->local_fz[i];
        }

        float vx = (sim->bodies[i].vx + fx * dt) * damping;
        float vy = (sim->bodies[i].vy + fy * dt) * damping;
        float vz = (sim->bodies[i].vz + fz * dt) * damping;

        float speed2 = vx * vx + vy * vy + vz * vz;
        if (speed2 > max_speed2) {
            float scale = max_speed / sqrtf(speed2);
            vx *= scale;
            vy *= scale;
            vz *= scale;
        }

        sim->bodies[i].vx = vx;
        sim->bodies[i].vy = vy;
        sim->bodies[i].vz = vz;
        sim->bodies[i].x += vx * dt;
        sim->bodies[i].y += vy * dt;
        sim->bodies[i].z += vz * dt;
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
            init_bodies_sphere(sim->bodies, sim->body_count, sqrtf((float)sim->body_count) * 20.0f);
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
    init_bodies_sphere(sim->bodies, body_count, sqrtf((float)body_count) * 20.0f);

    // Octree worst case is ~8N nodes; 16x is comfortably safe.
    sim->oct_storage_cap = body_count * 16 + 4096;
    sim->oct_storage = xmalloc((size_t)sim->oct_storage_cap * sizeof(*sim->oct_storage));

    for (int b = 0; b < 2; b++) {
        sim->draw_x[b] = xmalloc((size_t)body_count * sizeof(float));
        sim->draw_y[b] = xmalloc((size_t)body_count * sizeof(float));
        sim->draw_z[b] = xmalloc((size_t)body_count * sizeof(float));
    }
    sim->draw_index = 0;
    pthread_mutex_init(&sim->draw_mutex, NULL);

    for (int i = 0; i < body_count; i++) {
        for (int b = 0; b < 2; b++) {
            sim->draw_x[b][i] = sim->bodies[i].x;
            sim->draw_y[b][i] = sim->bodies[i].y;
            sim->draw_z[b][i] = sim->bodies[i].z;
        }
    }

    barrier_init(&sim->barrier, thread_count);

    sim->workers = xcalloc((size_t)thread_count, sizeof(*sim->workers));
    sim->threads = xmalloc((size_t)thread_count * sizeof(*sim->threads));

    for (int t = 0; t < thread_count; t++) {
        sim->workers[t].id = t;
        sim->workers[t].sim = sim;
        sim->workers[t].local_fx = xcalloc((size_t)body_count, sizeof(float));
        sim->workers[t].local_fy = xcalloc((size_t)body_count, sizeof(float));
        sim->workers[t].local_fz = xcalloc((size_t)body_count, sizeof(float));
        sim->workers[t].stack = xmalloc((size_t)sim->oct_storage_cap * sizeof(int));

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
        free(sim->workers[t].local_fz);
        free(sim->workers[t].stack);
    }

    barrier_destroy(&sim->barrier);
    pthread_mutex_destroy(&sim->draw_mutex);

    free(sim->threads);
    free(sim->workers);
    for (int b = 0; b < 2; b++) {
        free(sim->draw_x[b]);
        free(sim->draw_y[b]);
        free(sim->draw_z[b]);
    }
    free(sim->oct_storage);
    free(sim->bodies);
}

static int snapshot_draw_index(Sim *sim) {
    pthread_mutex_lock(&sim->draw_mutex);
    int idx = sim->draw_index;
    pthread_mutex_unlock(&sim->draw_mutex);
    return idx;
}

typedef struct {
    Vector3 target;
    float yaw;       // radians
    float pitch;     // radians, clamped near +/- pi/2
    float distance;
    float fov_deg;
} OrbitCamera;

static Camera3D orbit_to_camera(const OrbitCamera *oc) {
    Camera3D cam = {0};
    float cp = cosf(oc->pitch);
    float sp = sinf(oc->pitch);
    float cy = cosf(oc->yaw);
    float sy = sinf(oc->yaw);
    cam.position.x = oc->target.x + oc->distance * cp * sy;
    cam.position.y = oc->target.y + oc->distance * sp;
    cam.position.z = oc->target.z + oc->distance * cp * cy;
    cam.target = oc->target;
    cam.up = (Vector3){0.0f, 1.0f, 0.0f};
    cam.fovy = oc->fov_deg;
    cam.projection = CAMERA_PERSPECTIVE;
    return cam;
}

static OrbitCamera fit_orbit(const float *x, const float *y, const float *z, int n) {
    float minx = x[0], maxx = x[0];
    float miny = y[0], maxy = y[0];
    float minz = z[0], maxz = z[0];
    for (int i = 1; i < n; i++) {
        if (x[i] < minx) minx = x[i]; if (x[i] > maxx) maxx = x[i];
        if (y[i] < miny) miny = y[i]; if (y[i] > maxy) maxy = y[i];
        if (z[i] < minz) minz = z[i]; if (z[i] > maxz) maxz = z[i];
    }

    OrbitCamera oc;
    oc.target = (Vector3){0.5f * (minx + maxx), 0.5f * (miny + maxy), 0.5f * (minz + maxz)};
    oc.yaw = 0.6f;
    oc.pitch = 0.35f;
    oc.fov_deg = 60.0f;

    float bw = maxx - minx, bh = maxy - miny, bd = maxz - minz;
    float diag = sqrtf(bw * bw + bh * bh + bd * bd);
    if (diag < 1.0f) diag = 1.0f;
    float fov_rad = oc.fov_deg * (float)M_PI / 180.0f;
    oc.distance = 0.65f * diag / tanf(0.5f * fov_rad);
    return oc;
}

typedef struct {
    Matrix matVP;
    int sw, sh;
} Projector;

static Projector projector_make(Camera3D cam) {
    Projector p;
    p.sw = GetScreenWidth();
    p.sh = GetScreenHeight();

    Matrix matView = MatrixLookAt(cam.position, cam.target, cam.up);
    Matrix matProj = MatrixPerspective((double)cam.fovy * (M_PI / 180.0),
                                       (double)p.sw / (double)p.sh,
                                       0.01, 100000.0);
    // raylib's MatrixMultiply(L, R) yields R*L semantically, so this is matProj*matView.
    p.matVP = MatrixMultiply(matView, matProj);
    return p;
}

static inline int projector_project(const Projector *p, float x, float y, float z, int *sx, int *sy) {
    Matrix m = p->matVP;
    float w_clip = m.m3 * x + m.m7 * y + m.m11 * z + m.m15;
    if (w_clip <= 0.001f) return 0;  // behind near plane

    float x_clip = m.m0 * x + m.m4 * y + m.m8  * z + m.m12;
    float y_clip = m.m1 * x + m.m5 * y + m.m9  * z + m.m13;

    float invw = 1.0f / w_clip;
    float ndcx =  x_clip * invw;
    float ndcy = -y_clip * invw;  // raylib flips y after the divide

    *sx = (int)((ndcx + 1.0f) * 0.5f * (float)p->sw);
    *sy = (int)((ndcy + 1.0f) * 0.5f * (float)p->sh);
    return 1;
}

static inline unsigned u32_log2_floor(unsigned x) {
    unsigned r = 0;
    while (x >>= 1u) r++;
    return r;
}

static inline Color point_color_3d(int i, float xi, float yi, float zi, int scheme) {
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
            float d2 = xi * xi + yi * yi + zi * zi;
            unsigned bucket = (unsigned)(d2 * 0.00006f);
            return kPalette16[bucket & 15u];
        }

        case COLOR_SCHEME_XYZ_XOR: {
            unsigned ax = (unsigned)((int)(fabsf(xi) * 0.035f));
            unsigned ay = (unsigned)((int)(fabsf(yi) * 0.035f));
            unsigned az = (unsigned)((int)(fabsf(zi) * 0.035f));
            return kPalette16[(ax ^ ay ^ az) & 15u];
        }

        default:
            return (Color){123, 236, 178, 230};
    }
}

static void draw_points_3d(const float *x, const float *y, const float *z, int n,
                           const Projector *proj, int color_scheme, int chunky) {
    int sw = proj->sw;
    int sh = proj->sh;

    if (chunky) {
        for (int i = 1; i < n; i++) {
            int sx, sy;
            if (!projector_project(proj, x[i], y[i], z[i], &sx, &sy)) continue;
            if ((unsigned)sx < (unsigned)sw && (unsigned)sy < (unsigned)sh) {
                DrawRectangle(sx, sy, 2, 2, point_color_3d(i, x[i], y[i], z[i], color_scheme));
            }
        }
    } else {
        for (int i = 1; i < n; i++) {
            int sx, sy;
            if (!projector_project(proj, x[i], y[i], z[i], &sx, &sy)) continue;
            if ((unsigned)sx < (unsigned)sw && (unsigned)sy < (unsigned)sh) {
                DrawPixel(sx, sy, point_color_3d(i, x[i], y[i], z[i], color_scheme));
            }
        }
    }

    // Root marker
    int rsx, rsy;
    if (projector_project(proj, 0.0f, 0.0f, 0.0f, &rsx, &rsy)) {
        DrawCircle(rsx, rsy, 4.0f, (Color){236, 178, 123, 255});
    }
}

static void draw_edges_3d(const Edge *edges, int edge_count,
                          const float *x, const float *y, const float *z,
                          const Projector *proj) {
    Color edge_color = (Color){70, 80, 100, 22};
    for (int e = 0; e < edge_count; e++) {
        int a = edges[e].from;
        int b = edges[e].to;
        int sax, say, sbx, sby;
        if (!projector_project(proj, x[a], y[a], z[a], &sax, &say)) continue;
        if (!projector_project(proj, x[b], y[b], z[b], &sbx, &sby)) continue;
        DrawLine(sax, say, sbx, sby, edge_color);
    }
}

int main(int argc, char **argv) {
    int max_number = 1 << 20;
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
    InitWindow(1600, 1000, "Prime-rooted multiples graph - 3D realtime MT");
    SetTargetFPS(24);

    int idx = snapshot_draw_index(&sim);
    OrbitCamera oc = fit_orbit(sim.draw_x[idx], sim.draw_y[idx], sim.draw_z[idx], max_number);

    int draw_edges_flag = 0;
    int chunky_points = 0;
    int color_scheme = COLOR_SCHEME_MINT;
    long long last_steps = 0;
    double sim_sps_smoothed = 0.0;

    const float orbit_speed = 0.005f;
    const float pitch_lim = M_PI/2-0.00001f;

    while (!WindowShouldClose()) {    
        if (IsKeyPressed(KEY_SPACE))     sim.paused = !sim.paused;
        if (IsKeyPressed(KEY_E))         draw_edges_flag = !draw_edges_flag;
        if (IsKeyPressed(KEY_C))         color_scheme = (color_scheme + 1) % COLOR_SCHEME_COUNT;
        if (IsKeyPressed(KEY_P))         chunky_points = !chunky_points;
        if (IsKeyPressed(KEY_R)) {
            idx = snapshot_draw_index(&sim);
            oc = fit_orbit(sim.draw_x[idx], sim.draw_y[idx], sim.draw_z[idx], max_number);
        }
        if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD)) {
            if (sim.substeps_per_publish < 64) sim.substeps_per_publish *= 2;
        }
        if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT)) {
            if (sim.substeps_per_publish > 1) sim.substeps_per_publish /= 2;
        }
        if (IsKeyPressed(KEY_BACKSPACE)) sim.reset_requested = 1;

        // wheel zoom (orbit distance)
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f) {
            float factor = powf(0.88f, wheel);
            oc.distance *= factor;
            if (oc.distance < 0.5f)    oc.distance = 0.5f;
            if (oc.distance > 1.0e7f)  oc.distance = 1.0e7f;
        }

        // right-drag orbit
        if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
            Vector2 d = GetMouseDelta();
            oc.yaw   -= d.x * orbit_speed;
            oc.pitch -= d.y * orbit_speed;
            if (oc.pitch >  pitch_lim) oc.pitch =  pitch_lim;
            if (oc.pitch < -pitch_lim) oc.pitch = -pitch_lim;
        }

        // middle-drag pan (in screen-aligned plane through target)
        if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
            Vector2 d = GetMouseDelta();
            Camera3D cam = orbit_to_camera(&oc);
            Vector3 fwd   = Vector3Normalize(Vector3Subtract(cam.target, cam.position));
            Vector3 right = Vector3Normalize(Vector3CrossProduct(fwd, cam.up));
            Vector3 up    = Vector3Normalize(Vector3CrossProduct(right, fwd));
            float fov_rad = oc.fov_deg * (float)M_PI / 180.0f;
            float scale = oc.distance * 2.0f * tanf(0.5f * fov_rad) / (float)GetScreenHeight();

            oc.target.x -= d.x * scale * right.x;
            oc.target.y -= d.x * scale * right.y;
            oc.target.z -= d.x * scale * right.z;
            oc.target.x += d.y * scale * up.x;
            oc.target.y += d.y * scale * up.y;
            oc.target.z += d.y * scale * up.z;
        }

        // --- pull latest published positions ---
        idx = snapshot_draw_index(&sim);
        const float *x = sim.draw_x[idx];
        const float *y = sim.draw_y[idx];
        const float *z = sim.draw_z[idx];

        long long steps_now = sim.steps;
        double fps = (double)GetFPS();
        if (fps > 0.0) {
            double instant = (double)(steps_now - last_steps) * fps;
            sim_sps_smoothed = sim_sps_smoothed == 0.0 ? instant
                                                       : sim_sps_smoothed * 0.90 + instant * 0.10;
        }
        last_steps = steps_now;

        Camera3D cam = orbit_to_camera(&oc);
        Projector proj = projector_make(cam);

        // --- render ---
        BeginDrawing();
        ClearBackground((Color){8, 10, 14, 255});

        if (draw_edges_flag) draw_edges_3d(edges, edge_count, x, y, z, &proj);
        draw_points_3d(x, y, z, max_number, &proj, color_scheme, chunky_points);

        // HUD
        DrawRectangle(12, 12, 700, 204, (Color){0, 0, 0, 155});
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
        DrawText(TextFormat("color: %s | points: %s | yaw: %+.2f pitch: %+.2f dist: %.0f",
                            kColorSchemeNames[color_scheme],
                            chunky_points ? "2x2" : "1px",
                            oc.yaw, oc.pitch, oc.distance),
                 24, 168, 20, RAYWHITE);

        DrawText("right-drag orbit | middle-drag pan | wheel zoom | space pause | e edges | c colors | p point size | r refit | backspace reset | +/- publish stride",
                 24, GetScreenHeight() - 32, 16, GRAY);

        EndDrawing();
    }

    sim_stop(&sim);
    free(edges);

    CloseWindow();
    return 0;
}
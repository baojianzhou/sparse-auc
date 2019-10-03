//
// Created by baojian on 9/9/19.
//
#include "auc_opt_methods.h"


void randn(int n, double *samples) {
    for (int i = 0; i < n; i++) {
        double x = (double) random() / (RAND_MAX * 1.);
        double y = (double) random() / (RAND_MAX * 1.);
        samples[i] = sqrt(-2.0 * log(x)) * cos(2.0 * M_PI * y);
    }
}

static inline int __comp_descend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return 1;
    } else {
        return -1;
    }
}

void _arg_sort_descend(const double *x, int *sorted_indices, int x_len) {
    data_pair *w_pairs = malloc(sizeof(data_pair) * x_len);
    for (int i = 0; i < x_len; i++) {
        w_pairs[i].val = x[i];
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) x_len, sizeof(data_pair), &__comp_descend);
    for (int i = 0; i < x_len; i++) {
        sorted_indices[i] = w_pairs[i].index;
    }
    free(w_pairs);
}

GraphStat *make_graph_stat(int p, int m) {
    GraphStat *stat = malloc(sizeof(GraphStat));
    stat->num_pcst = 0;
    stat->re_edges = malloc(sizeof(Array));
    stat->re_edges->size = 0;
    stat->re_edges->array = malloc(sizeof(int) * p);
    stat->re_nodes = malloc(sizeof(Array));
    stat->re_nodes->size = 0;
    stat->re_nodes->array = malloc(sizeof(int) * p);
    stat->run_time = 0;
    stat->costs = malloc(sizeof(double) * m);
    stat->prizes = malloc(sizeof(double) * p);
    return stat;
}

bool free_graph_stat(GraphStat *graph_stat) {
    free(graph_stat->re_nodes->array);
    free(graph_stat->re_nodes);
    free(graph_stat->re_edges->array);
    free(graph_stat->re_edges);
    free(graph_stat->costs);
    free(graph_stat->prizes);
    free(graph_stat);
    return true;
}

// compare function for descending sorting.
int _comp_descend(const void *a, const void *b) {
    if (((data_pair *) a)->val < ((data_pair *) b)->val) {
        return 1;
    } else {
        return -1;
    }
}

// get positive minimum prize in prizes vector.
double min_pi(
        const double *prizes, double *total_prizes, int n, double err_tol,
        int verbose) {
    *total_prizes = 0.0;
    double positive_min = INFINITY;
    for (int ii = 0; ii < n; ii++) {
        *total_prizes += prizes[ii];
        if ((prizes[ii] < positive_min) && (prizes[ii] > 0.0)) {
            positive_min = prizes[ii];
        }
    }
    /**
     * Warning: There is a precision issue here. We may need to define a
     * minimum precision. In our experiment,  we found that some very
     * small positive number could be like 1.54046e-310, 1.54046e-310.
     * In this case, the fast-pcst cannot stop!!!
     */
    if (positive_min < err_tol) {
        if (verbose > 0) {
            printf("warning too small positive val found.\n");
        }
        positive_min = err_tol;
    }
    return positive_min;
}

// deep first search for finding a tour.
bool dfs_tour(
        const EdgePair *edges, int n, Array *tree_nodes, Array *tree_edges,
        Array *tour_nodes, Array *tour_edges) {
    /**
     * This method is to find a euler tour given a tree. This method is
     * proposed in the following paper:
     *  Authors : Edmonds, Jack, and Ellis L. Johnson.
     *  Title : Matching, Euler tours and the Chinese postman
     *  Journal: Mathematical programming 5.1 (1973): 88-124.
     */
    //Make sure the tree has at least two nodes.
    if (tree_nodes->size <= 1) {
        printf("error: The input tree has at least two nodes.");
        exit(0);
    }
    typedef struct {
        int first;
        int second;
        bool third;
    } Tuple;
    typedef struct {
        Tuple *array;
        size_t size;
    } Nei;
    int i, *nei_size = calloc((size_t) n, sizeof(int));
    for (i = 0; i < tree_edges->size; i++) {
        nei_size[edges[tree_edges->array[i]].first]++;
        nei_size[edges[tree_edges->array[i]].second]++;
    }
    Nei *adj = malloc(sizeof(Nei) * n);
    for (i = 0; i < n; i++) {
        adj[i].size = 0;
        adj[i].array = malloc(sizeof(Tuple) * nei_size[i]);
    }
    for (i = 0; i < tree_edges->size; i++) {
        // each tuple is: (indexed node, edge_index, is_visited)
        int uu = edges[tree_edges->array[i]].first;
        int vv = edges[tree_edges->array[i]].second;
        Tuple nei_v, nei_u;
        nei_v.second = tree_edges->array[i];
        nei_u.second = tree_edges->array[i];
        nei_v.first = vv, nei_v.third = false;
        nei_u.first = uu, nei_u.third = false;
        adj[uu].array[adj[uu].size++] = nei_v;  // edge u --> v
        adj[vv].array[adj[vv].size++] = nei_u;  // edge v --> u
    }
    // The first element as tour's root.
    int start_node = tree_nodes->array[0];
    bool *visited = calloc((size_t) n, sizeof(bool));
    tour_nodes->array[tour_nodes->size++] = start_node;
    while (true) {
        bool flag_1 = false;
        visited[start_node] = true;
        // iterate the adj of each node. in this loop, we check if
        // there exists any its neighbor which has not been visited.
        for (i = 0; i < (int) adj[start_node].size; i++) {
            int next_node = adj[start_node].array[i].first;
            int edge_index = adj[start_node].array[i].second;
            if (!visited[next_node]) { // first time to visit this node.
                visited[next_node] = true; // mark it as visited.
                tour_nodes->array[tour_nodes->size++] = next_node;
                tour_edges->array[tour_edges->size++] = edge_index;
                adj[start_node].array[i].third = true; // mark it as labeled.
                start_node = next_node;
                flag_1 = true;
                break;
            }
        }
        // all neighbors are visited. Then we check if
        // there exists adj which is false nodes.
        if (!flag_1) {
            bool flag_2 = false;
            for (i = 0; i < (int) adj[start_node].size; i++) {
                int next_node = adj[start_node].array[i].first;
                int edge_index = adj[start_node].array[i].second;
                bool is_visited = adj[start_node].array[i].third;
                // there exists a neighbor. has false node
                if (!is_visited) {
                    adj[start_node].array[i].third = true;
                    tour_nodes->array[tour_nodes->size++] = next_node;
                    tour_edges->array[tour_edges->size++] = edge_index;
                    start_node = next_node;
                    flag_2 = true;
                    break;
                }
            }
            // all nodes are visited and there is no false nodes.
            if (!flag_2) {
                break;
            }
        }
    }
    free(visited);
    for (i = 0; i < n; i++) { free(adj[i].array); }
    free(adj);
    free(nei_size);
    return true;
}

// find a dense tree
bool prune_tree(
        const EdgePair *edges, const double *prizes, const double *costs,
        int n, int m, double c_prime, Array *tree_nodes, Array *tree_edges) {
    Array *tour_nodes = malloc(sizeof(Array));
    Array *tour_edges = malloc(sizeof(Array));
    tour_nodes->size = 0, tour_edges->size = 0;
    tour_nodes->array = malloc(sizeof(int) * (2 * tree_nodes->size - 1));
    tour_edges->array = malloc(sizeof(int) * (2 * tree_nodes->size - 2));
    dfs_tour(edges, n, tree_nodes, tree_edges, tour_nodes, tour_edges);
    // calculating pi_prime.
    double *pi_prime = malloc(sizeof(double) * (2 * tour_nodes->size - 1));
    int pi_prime_size = 0;
    bool *tmp_vector = calloc((size_t) n, sizeof(bool));
    for (int ii = 0; ii < tour_nodes->size; ii++) {
        // first time show in the tour.
        if (!tmp_vector[tour_nodes->array[ii]]) {
            pi_prime[pi_prime_size++] = prizes[tour_nodes->array[ii]];
            tmp_vector[tour_nodes->array[ii]] = true;
        } else {
            pi_prime[pi_prime_size++] = 0.0;
        }
    }
    double prize_t = 0.0, cost_t = 0.0;
    for (int ii = 0; ii < tree_nodes->size; ii++) {
        prize_t += prizes[tree_nodes->array[ii]];
    }
    for (int ii = 0; ii < tree_edges->size; ii++) {
        cost_t += costs[tree_edges->array[ii]];
    }
    tree_nodes->size = 0;
    tree_edges->size = 0;
    double phi = prize_t / cost_t;
    for (int ii = 0; ii < tour_nodes->size; ii++) {
        if (prizes[tour_nodes->array[ii]] >= ((c_prime * phi) / 6.)) {
            // create a single node tree.
            tree_nodes->array[tree_nodes->size++] = tour_nodes->array[ii];
            free(tmp_vector);
            free(pi_prime);
            free(tour_nodes->array);
            free(tour_edges->array);
            free(tour_nodes);
            free(tour_edges);
            return true;
        }
    }
    Array *p_l = malloc(sizeof(Array));
    p_l->size = 0;
    p_l->array = malloc(sizeof(int) * (2 * tour_nodes->size - 1));
    for (int i = 0; i < tour_nodes->size; i++) {
        p_l->array[p_l->size++] = i;
        double pi_prime_pl = 0.0;
        for (int ii = 0; ii < p_l->size; ii++) {
            pi_prime_pl += pi_prime[p_l->array[ii]];
        }
        double c_prime_pl = 0.0;
        if (p_l->size >= 2) { // <= 1: there is no edge.
            for (int j = 0; j < p_l->size - 1; j++) {
                c_prime_pl += costs[tour_edges->array[p_l->array[j]]];
            }
        }
        if (c_prime_pl > c_prime) { // start a new sublist
            p_l->size = 0;
        } else if (pi_prime_pl >= ((c_prime * phi) / 6.)) {
            bool *added_nodes = calloc((size_t) n, sizeof(bool));
            bool *added_edges = calloc((size_t) m, sizeof(bool));
            for (int j = 0; j < p_l->size; j++) {
                int cur_node = tour_nodes->array[p_l->array[j]];
                if (!added_nodes[cur_node]) {
                    added_nodes[cur_node] = true;
                    tree_nodes->array[tree_nodes->size++] = cur_node;
                }
                int cur_edge = tour_edges->array[p_l->array[j]];
                if (!added_edges[cur_edge]) {
                    added_edges[cur_edge] = true;
                    tree_edges->array[tree_edges->size++] = cur_edge;
                }
            }
            tree_edges->size--; // pop the last edge
            free(added_edges);
            free(added_nodes);
            free(tmp_vector);
            free(pi_prime);
            free(tour_nodes->array);
            free(tour_edges->array);
            free(tour_nodes);
            free(tour_edges);
            free(p_l->array);
            free(p_l);
            return true;
        }
    }
    printf("Error: Never reach at this point.\n"); //Merge procedure.
    exit(0);
}


//sort g trees in the forest
Tree *sort_forest(
        const EdgePair *edges, const double *prizes, const double *costs,
        int g, int n, Array *f_nodes, Array *f_edges, int *sorted_ind) {
    typedef struct {
        int *array;
        size_t size;
    } Nei;
    int *neighbors_size = calloc((size_t) n, sizeof(int));
    Nei *adj = malloc(sizeof(Nei) * n);
    for (int ii = 0; ii < f_edges->size; ii++) {
        neighbors_size[edges[f_edges->array[ii]].first]++;
        neighbors_size[edges[f_edges->array[ii]].second]++;
    }
    for (int ii = 0; ii < n; ii++) {
        adj[ii].size = 0;
        adj[ii].array = malloc(sizeof(int) * neighbors_size[ii]);
    }
    for (int ii = 0; ii < f_edges->size; ii++) {
        int uu = edges[f_edges->array[ii]].first;
        int vv = edges[f_edges->array[ii]].second;
        adj[uu].array[adj[uu].size++] = vv;  // edge u --> v
        adj[vv].array[adj[vv].size++] = uu;  // edge v --> u
    }
    int t = 0; // component id
    //label nodes to the components id.
    int *comp = calloc((size_t) n, sizeof(int));
    bool *visited = calloc((size_t) n, sizeof(bool));
    int *stack = malloc(sizeof(int) * n), stack_size = 0;
    int *comp_size = calloc((size_t) g, sizeof(int));
    // dfs algorithm to get cc
    for (int ii = 0; ii < f_nodes->size; ii++) {
        int cur_node = f_nodes->array[ii];
        if (!visited[cur_node]) {
            stack[stack_size++] = cur_node;
            while (stack_size != 0) { // check empty
                int s = stack[stack_size - 1];
                stack_size--;
                if (!visited[s]) {
                    visited[s] = true;
                    comp[s] = t;
                    comp_size[t]++;
                }
                for (int k = 0; k < (int) adj[s].size; k++) {
                    if (!visited[adj[s].array[k]]) {
                        stack[stack_size++] = adj[s].array[k];
                    }
                }
            }
            t++; // to label component id.
        }
    }
    Tree *trees = malloc(sizeof(Tree) * g);
    for (int ii = 0; ii < g; ii++) {
        int tree_size = comp_size[ii];
        trees[ii].nodes = malloc(sizeof(Array));
        trees[ii].edges = malloc(sizeof(Array));
        trees[ii].nodes->size = 0;
        trees[ii].edges->size = 0;
        trees[ii].nodes->array = malloc(sizeof(int) * tree_size);
        trees[ii].edges->array = malloc(sizeof(int) * (tree_size - 1));
        trees[ii].prize = 0.0;
        trees[ii].cost = 0.0;
    }
    // insert nodes into trees.
    for (int ii = 0; ii < f_nodes->size; ii++) {
        int tree_i = comp[f_nodes->array[ii]];
        int cur_node = f_nodes->array[ii];
        trees[tree_i].nodes->array[trees[tree_i].nodes->size++] = cur_node;
        trees[tree_i].prize += prizes[f_nodes->array[ii]];
    }
    // insert edges into trees.
    for (int ii = 0; ii < f_edges->size; ii++) {
        // random select one endpoint
        int uu = edges[f_edges->array[ii]].first;
        int tree_i = comp[uu], cur_edge = f_edges->array[ii];
        trees[tree_i].edges->array[trees[tree_i].edges->size++] = cur_edge;
        trees[tree_i].cost += costs[f_edges->array[ii]];
    }

    data_pair *w_pairs = (data_pair *) malloc(sizeof(data_pair) * g);
    for (int i = 0; i < g; i++) {
        if (trees[i].cost > 0.0) { // tree weight
            w_pairs[i].val = trees[i].prize / trees[i].cost;
        } else { // for a single node tree
            w_pairs[i].val = INFINITY;
        }
        w_pairs[i].index = i;
    }
    qsort(w_pairs, (size_t) g, sizeof(data_pair), &_comp_descend);
    for (int i = 0; i < g; i++) {
        sorted_ind[i] = w_pairs[i].index;
    }
    free(w_pairs);
    free(comp_size);
    free(stack);
    free(visited);
    free(comp);
    for (int ii = 0; ii < n; ii++) { free(adj[ii].array); }
    free(adj);
    free(neighbors_size);
    return trees;
}

bool prune_forest(
        const EdgePair *edges, const double *prizes, const double *costs,
        int g, int n, int m, double C, Array *f_nodes, Array *f_edges) {
    // case 1: usually, there is only one tree. then forest is a tree.
    int i, j;
    double cost_f = 0.0, prize_f = 0.0;
    for (i = 0; i < f_nodes->size; i++) {
        prize_f += prizes[f_nodes->array[i]];
    }
    for (i = 0; i < f_edges->size; i++) {
        cost_f += costs[f_edges->array[i]];
    }
    if (g == 1) {
        // single node forest or it is already good enough
        if (cost_f <= C) {
            return true;
        } else if (0.0 < C) {
            // must have at least two nodes
            prune_tree(edges, prizes, costs, n, m, C, f_nodes, f_edges);
            return true;
        } else {
            //return a maximal node
            int max_node = f_nodes->array[0];
            double max_prize = prizes[max_node];
            for (i = 0; i < f_nodes->size; i++) {
                if (max_prize < prizes[f_nodes->array[i]]) {
                    max_prize = prizes[f_nodes->array[i]];
                    max_node = f_nodes->array[i];
                }
            }
            f_nodes->size = 1;
            f_nodes->array[0] = max_node;
            f_edges->size = 0;
            return true;
        }
    }
    // case 2: there are at least two trees.
    int *sorted_ind = malloc(sizeof(int) * g);
    Tree *trees;
    trees = sort_forest(
            edges, prizes, costs, g, n, f_nodes, f_edges, sorted_ind);
    //clear nodes_f and edges_f, and then update them.
    f_nodes->size = 0, f_edges->size = 0;
    double c_r = C;
    for (i = 0; i < g; i++) {
        int sorted_i = sorted_ind[i];
        double c_tree_i = trees[sorted_i].cost;
        if (c_r >= c_tree_i) {
            c_r -= c_tree_i;
        } else if (c_r > 0.0) {
            // tree_i must have at least two nodes and one edge.
            prune_tree(edges, prizes, costs, n, m, c_r,
                       trees[sorted_i].nodes, trees[sorted_i].edges);
            c_r = 0.0;
        } else {
            // get maximal node
            int max_node = trees[sorted_i].nodes->array[0];
            double max_prize = prizes[max_node];
            for (int ii = 0; ii < trees[sorted_i].nodes->size; ii++) {
                if (max_prize < prizes[trees[sorted_i].nodes->array[ii]]) {
                    max_prize = prizes[trees[sorted_i].nodes->array[ii]];
                    max_node = trees[sorted_i].nodes->array[ii];
                }
            }
            trees[sorted_i].nodes->size = 1;
            trees[sorted_i].nodes->array[0] = max_node;
            trees[sorted_i].edges->size = 0;
        }
        for (j = 0; j < trees[sorted_i].nodes->size; j++) {
            int cur_node = trees[sorted_i].nodes->array[j];
            f_nodes->array[f_nodes->size++] = cur_node;
        }
        for (j = 0; j < trees[sorted_i].edges->size; j++) {
            int cur_edge = trees[sorted_i].edges->array[j];
            f_edges->array[f_edges->size++] = cur_edge;
        }
    }// iterate trees by descending order.
    for (i = 0; i < g; i++) { free(trees[i].nodes), free(trees[i].edges); }
    free(trees), free(sorted_ind);
    printf("pruning forest\n****\n");
    return true;
}


bool head_proj_exact(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double delta, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat) {
    stat->re_nodes->size = 0, stat->re_edges->size = 0,
    stat->run_time = 0.0, stat->num_pcst = 0;
    PCST *pcst;
    clock_t start_time = clock();
    double total_prizes;
    double *tmp_prizes = malloc(sizeof(double) * n);
    // total_prizes will be calculated.
    double pi_min = min_pi(prizes, &total_prizes, n, err_tol, verbose);
    double lambda_r = (2. * C) / (pi_min);
    double lambda_l = 1. / (4. * total_prizes);
    double lambda_m;
    double epsilon_ = (delta * C) / (2. * total_prizes);
    double cost_f;
    int i;
    for (i = 0; i < n; i++) { tmp_prizes[i] = prizes[i] * lambda_r; }
    pcst = make_pcst(edges, tmp_prizes, costs, root, g,
                     epsilon, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    stat->num_pcst++;
    cost_f = 0.0;
    for (i = 0; i < stat->re_edges->size; i++) {
        cost_f += costs[stat->re_edges->array[i]];
    }
    if (cost_f <= (2. * C)) {
        stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
        return true;
    }// ensure that we have invariant c(F_r) > 2 C
    while ((lambda_r - lambda_l) > epsilon_) {
        lambda_m = (lambda_l + lambda_r) / 2.;
        for (i = 0; i < n; i++) { tmp_prizes[i] = prizes[i] * lambda_m; }
        pcst = make_pcst(edges, tmp_prizes, costs, root, g,
                         epsilon, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
        stat->num_pcst++;
        cost_f = 0.0;
        for (i = 0; i < stat->re_edges->size; i++) {
            cost_f += costs[stat->re_edges->array[i]];
        }
        if (cost_f > (2. * C)) {
            lambda_r = lambda_m;
        } else {
            lambda_l = lambda_m;
        }
        if (stat->num_pcst >= max_iter) {
            if (verbose > 0) {
                printf("Warn(head): number iteration is beyond max_iter.\n");
            }
            break;
        }
    } // binary search over the Lagrange parameter lambda
    Array *l_re_nodes = malloc(sizeof(Array));
    Array *l_re_edges = malloc(sizeof(Array));
    l_re_nodes->array = malloc(sizeof(int) * n);
    l_re_edges->array = malloc(sizeof(int) * n);
    for (i = 0; i < n; i++) { tmp_prizes[i] = prizes[i] * lambda_l; }
    pcst = make_pcst(edges, tmp_prizes, costs, root, g,
                     epsilon, pruning, n, m, verbose);
    run_pcst(pcst, l_re_nodes, l_re_edges), free_pcst(pcst);
    stat->num_pcst++;
    Array *r_re_nodes = malloc(sizeof(Array));
    Array *r_re_edges = malloc(sizeof(Array));
    r_re_nodes->array = malloc(sizeof(int) * n);
    r_re_edges->array = malloc(sizeof(int) * n);
    for (i = 0; i < n; i++) { tmp_prizes[i] = prizes[i] * lambda_r; }
    pcst = make_pcst(edges, tmp_prizes, costs, root, g,
                     epsilon, pruning, n, m, verbose);
    run_pcst(pcst, r_re_nodes, r_re_edges), free_pcst(pcst);
    stat->num_pcst++;
    prune_forest(edges, prizes, costs, g, n, m, C, r_re_nodes, r_re_edges);
    double l_prize_f = 0.0, r_prize_f = 0.0;
    for (i = 0; i < l_re_nodes->size; i++) {
        l_prize_f += prizes[l_re_nodes->array[i]];
    }
    for (i = 0; i < r_re_nodes->size; i++) {
        r_prize_f += prizes[r_re_nodes->array[i]];
    }
    if (l_prize_f >= r_prize_f) { //get the left one
        stat->re_nodes->size = l_re_nodes->size;
        for (i = 0; i < stat->re_nodes->size; i++) {
            stat->re_nodes->array[i] = l_re_nodes->array[i];
        }
        stat->re_edges->size = l_re_edges->size;
        for (i = 0; i < stat->re_edges->size; i++) {
            stat->re_edges->array[i] = l_re_edges->array[i];
        }
    } else { // get the right one
        stat->re_nodes->size = r_re_nodes->size;
        for (i = 0; i < stat->re_nodes->size; i++) {
            stat->re_nodes->array[i] = r_re_nodes->array[i];
        }
        stat->re_edges->size = r_re_edges->size;
        for (i = 0; i < stat->re_edges->size; i++) {
            stat->re_edges->array[i] = r_re_edges->array[i];
        }
    }
    stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    free(l_re_nodes->array), free(l_re_edges->array);
    free(r_re_nodes->array), free(r_re_edges->array);
    free(l_re_nodes), free(l_re_edges);
    free(r_re_nodes), free(r_re_edges);
    free(tmp_prizes);
    return true;
}

bool head_proj_approx(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double delta, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat) {
    return head_proj_exact(
            edges, costs, prizes, g, C, delta, max_iter, err_tol, root,
            pruning, epsilon, n, m, verbose, stat);
}

bool tail_proj_exact(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double nu, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat) {
    stat->re_nodes->size = 0, stat->re_edges->size = 0,
    stat->run_time = 0.0, stat->num_pcst = 0;
    clock_t start_time = clock();
    int i;
    double total_prizes = 0.0, c_f = 0.0, pi_f_bar = 0.0;
    double pi_min = min_pi(prizes, &total_prizes, n, err_tol, verbose);
    double lambda_0 = pi_min / (2.0 * C);
    double *tmp_costs = malloc(sizeof(double) * m);
    for (i = 0; i < m; i++) { tmp_costs[i] = costs[i] * lambda_0; }
    PCST *pcst;
    pcst = make_pcst(edges, prizes, tmp_costs, root, g,
                     epsilon, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    stat->num_pcst++;
    for (i = 0; i < stat->re_edges->size; i++) {
        c_f += costs[stat->re_edges->array[i]];
    }
    for (i = 0; i < stat->re_nodes->size; i++) {
        pi_f_bar += prizes[stat->re_nodes->array[i]];
    }
    pi_f_bar = total_prizes - pi_f_bar;
    if ((c_f <= (2.0 * C)) && (pi_f_bar <= 0.0)) {
        free(tmp_costs);
        stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
        return true;
    }
    double lambda_r = 0., lambda_l = 3. * total_prizes, lambda_m;
    double epsilon_ = (pi_min * fmin(0.5, 1. / nu)) / C;
    while ((lambda_l - lambda_r) > epsilon_) {
        lambda_m = (lambda_l + lambda_r) / 2.;
        for (i = 0; i < m; i++) { tmp_costs[i] = costs[i] * lambda_m; }
        pcst = make_pcst(edges, prizes, tmp_costs, root, g,
                         epsilon, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
        stat->num_pcst++;
        c_f = 0.0;
        for (i = 0; i < stat->re_edges->size; i++) {
            c_f += costs[stat->re_edges->array[i]];
        }
        if ((c_f >= (2. * C)) && (c_f <= (nu * C))) {
            free(tmp_costs);
            stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
            return true;
        }
        if (c_f >= (nu * C)) {
            lambda_r = lambda_m;
        } else {
            lambda_l = lambda_m;
        }
        if (stat->num_pcst >= max_iter) {
            if (verbose > 0) {
                printf("Warn(tail): number iteration is beyond max_iter.\n");
            }
            break;
        }
    } // end while
    for (int ii = 0; ii < m; ii++) {
        tmp_costs[ii] = costs[ii] * lambda_l;
    }
    pcst = make_pcst(edges, prizes, tmp_costs, root, g,
                     epsilon, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    stat->num_pcst++;
    free(tmp_costs);
    stat->run_time = (double) (clock() - start_time) / CLOCKS_PER_SEC;
    return true;
}


bool tail_proj_approx(
        const EdgePair *edges, const double *costs, const double *prizes,
        int g, double C, double nu, int max_iter, double err_tol, int root,
        PruningMethod pruning, double epsilon, int n, int m, int verbose,
        GraphStat *stat) {
    return tail_proj_exact(
            edges, costs, prizes, g, C, nu, max_iter, err_tol, root, pruning,
            epsilon, n, m, verbose, stat);
}


bool cluster_grid_pcst(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, double lambda,
        int root, PruningMethod pruning, int verbose,
        GraphStat *stat) {
    double *costs_ = malloc(sizeof(double) * m);
    double *prizes_ = malloc(sizeof(double) * n);
    for (int ii = 0; ii < m; ii++) {
        costs_[ii] = costs[ii] * lambda;
    }
    for (int ii = 0; ii < n; ii++) {
        prizes_[ii] = prizes[ii];
    }
    PCST *pcst;
    pcst = make_pcst(edges, prizes_, costs_, root, target_num_clusters,
                     1e-10, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    return true;

}

bool cluster_grid_pcst_binsearch(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, int root, int sparsity_low,
        int sparsity_high, int max_num_iter, PruningMethod pruning,
        int verbose, GraphStat *stat) {
    double *cur_costs = malloc(sizeof(double) * m);
    for (int ii = 0; ii < m; ii++) {
        cur_costs[ii] = costs[ii];
    }


    double *sorted_prizes = malloc(sizeof(double) * n);
    int *sorted_indices = malloc(sizeof(int) * n);
    for (int ii = 0; ii < n; ii++) {
        sorted_prizes[ii] = prizes[ii];
    }
    int guess_pos = n - sparsity_high;
    _arg_sort_descend(sorted_prizes, sorted_indices, n);
    double lambda_low = 0.0;
    double lambda_high = 2.0 * sorted_prizes[sorted_indices[guess_pos]];
    bool using_sparsity_low = false;
    bool using_max_value = false;
    if (lambda_high == 0.0) {
        guess_pos = n - sparsity_low;
        lambda_high = 2.0 * sorted_prizes[sorted_indices[guess_pos]];
        if (lambda_high != 0.0) {
            using_sparsity_low = true;
        } else {
            using_max_value = true;
            lambda_high = prizes[0];
            for (int ii = 1; ii < n; ii++) {
                lambda_high = fmax(lambda_high, prizes[ii]);
            }
            lambda_high *= 2.0;
        }
    }
    if (verbose >= 1) {
        const char *sparsity_low_text = "k_low";
        const char *sparsity_high_text = "k_high";
        const char *max_value_text = "max value";
        const char *guess_text = sparsity_high_text;
        if (using_sparsity_low) {
            guess_text = sparsity_low_text;
        } else if (using_max_value) {
            guess_text = max_value_text;
        }
        printf("n = %d  c: %d  k_low: %d  k_high: %d  l_low: %e  l_high: %e  "
               "max_num_iter: %d  (using %s for initial guess).\n",
               n, target_num_clusters, sparsity_low, sparsity_high,
               lambda_low, lambda_high, max_num_iter, guess_text);
    }
    int num_iter = 0;
    lambda_high /= 2.0;
    PCST *pcst;
    int cur_k;
    do {
        num_iter += 1;
        lambda_high *= 2.0;
        for (int ii = 0; ii < m; ++ii) {
            cur_costs[ii] = lambda_high * costs[ii];
        }
        pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                         1e-10, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) {
            printf("increase:   l_high: %e  k: %d\n", lambda_high, cur_k);
        }
    } while (cur_k > sparsity_high && num_iter < max_num_iter);

    if (num_iter < max_num_iter && cur_k >= sparsity_low) {
        if (verbose >= 1) {
            printf("Found good lambda in exponential "
                   "increase phase, returning.\n");
        }
        return true;
    }
    double lambda_mid;
    while (num_iter < max_num_iter) {
        num_iter += 1;
        lambda_mid = (lambda_low + lambda_high) / 2.0;
        for (int ii = 0; ii < m; ++ii) {
            cur_costs[ii] = lambda_mid * costs[ii];
        }
        pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                         1e-10, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) {
            printf("bin_search: l_mid:  %e  k: %d  "
                   "(lambda_low: %e  lambda_high: %e)\n", lambda_mid, cur_k,
                   lambda_low, lambda_high);
        }
        if (cur_k <= sparsity_high && cur_k >= sparsity_low) {
            if (verbose >= 1) {
                printf("Found good lambda in binary "
                       "search phase, returning.\n");
            }
            return true;
        }
        if (cur_k > sparsity_high) {
            lambda_low = lambda_mid;
        } else {
            lambda_high = lambda_mid;
        }
    }
    for (int ii = 0; ii < m; ++ii) {
        cur_costs[ii] = lambda_high * costs[ii];
    }
    pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                     1e-10, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges), free_pcst(pcst);
    if (verbose >= 1) {
        printf("Reached the maximum number of "
               "iterations, using the last l_high: %e  k: %d\n",
               lambda_high, stat->re_nodes->size);
    }
    return true;
}


bool head_tail_binsearch(
        const EdgePair *edges, const double *costs, const double *prizes,
        int n, int m, int target_num_clusters, int root, int sparsity_low,
        int sparsity_high, int max_num_iter, PruningMethod pruning,
        int verbose, GraphStat *stat) {

    // malloc: cur_costs, sorted_prizes, and sorted_indices
    // free: cur_costs, sorted_prizes, and sorted_indices
    double *cur_costs = malloc(sizeof(double) * m);
    double *sorted_prizes = malloc(sizeof(double) * n);
    int *sorted_indices = malloc(sizeof(int) * n);
    for (int ii = 0; ii < m; ii++) {
        cur_costs[ii] = costs[ii];
    }
    for (int ii = 0; ii < n; ii++) {
        sorted_prizes[ii] = prizes[ii];
    }
    int guess_pos = n - sparsity_high;
    _arg_sort_descend(sorted_prizes, sorted_indices, n);
    double lambda_low = 0.0;
    double lambda_high = 2.0 * sorted_prizes[sorted_indices[guess_pos]];
    bool using_sparsity_low = false;
    bool using_max_value = false;
    if (lambda_high == 0.0) {
        guess_pos = n - sparsity_low;
        lambda_high = 2.0 * sorted_prizes[sorted_indices[guess_pos]];
        if (lambda_high != 0.0) {
            using_sparsity_low = true;
        } else {
            using_max_value = true;
            lambda_high = prizes[0];
            for (int ii = 1; ii < n; ii++) {
                lambda_high = fmax(lambda_high, prizes[ii]);
            }
            lambda_high *= 2.0;
        }
    }
    if (verbose >= 1) {
        const char *sparsity_low_text = "k_low";
        const char *sparsity_high_text = "k_high";
        const char *max_value_text = "max value";
        const char *guess_text = sparsity_high_text;
        if (using_sparsity_low) {
            guess_text = sparsity_low_text;
        } else if (using_max_value) {
            guess_text = max_value_text;
        }
        printf("n = %d  c: %d  k_low: %d  k_high: %d  l_low: %e  l_high: %e  "
               "max_num_iter: %d  (using %s for initial guess).\n",
               n, target_num_clusters, sparsity_low, sparsity_high,
               lambda_low, lambda_high, max_num_iter, guess_text);
    }
    stat->num_iter = 0;
    lambda_high /= 2.0;
    int cur_k;
    do {
        stat->num_iter += 1;
        lambda_high *= 2.0;
        for (int ii = 0; ii < m; ii++) {
            cur_costs[ii] = lambda_high * costs[ii];
        }
        if (verbose >= 1) {
            for (int ii = 0; ii < m; ii++) {
                printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second,
                       cur_costs[ii]);
            }
            for (int ii = 0; ii < n; ii++) {
                printf("N %d %.15f\n", ii, prizes[ii]);
            }
            printf("\n");
            printf("lambda_high: %f\n", lambda_high);
            printf("target_num_clusters: %d\n", target_num_clusters);
        }
        PCST *pcst = make_pcst(
                edges, prizes, cur_costs, root, target_num_clusters,
                1e-10, pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges);
        free_pcst(pcst);
        cur_k = stat->re_nodes->size;

        if (verbose >= 1) {
            printf("increase:   l_high: %e  k: %d\n", lambda_high, cur_k);
        }
    } while (cur_k > sparsity_high && stat->num_iter < max_num_iter);

    if (stat->num_iter < max_num_iter && cur_k >= sparsity_low) {
        if (verbose >= 1) {
            printf("Found good lambda in exponential "
                   "increase phase, returning.\n");
        }
        free(cur_costs);
        free(sorted_prizes);
        free(sorted_indices);
        return true;
    }
    double lambda_mid;
    while (stat->num_iter < max_num_iter) {
        stat->num_iter += 1;
        lambda_mid = (lambda_low + lambda_high) / 2.0;
        for (int ii = 0; ii < m; ii++) {
            cur_costs[ii] = lambda_mid * costs[ii];
        }
        PCST *pcst = make_pcst(
                edges, prizes, cur_costs, root, target_num_clusters, 1e-10,
                pruning, n, m, verbose);
        run_pcst(pcst, stat->re_nodes, stat->re_edges);
        free_pcst(pcst);
        cur_k = stat->re_nodes->size;
        if (verbose >= 1) {
            for (int ii = 0; ii < m; ii++) {
                printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second,
                       cur_costs[ii]);
            }
            for (int ii = 0; ii < n; ii++) {
                printf("N %d %.15f\n", ii, prizes[ii]);
            }
            printf("bin_search: l_mid:  %e  k: %d  "
                   "(lambda_low: %e  lambda_high: %e)\n", lambda_mid, cur_k,
                   lambda_low, lambda_high);
        }
        if (sparsity_low <= cur_k && cur_k <= sparsity_high) {
            if (verbose >= 1) {
                printf("Found good lambda in binary "
                       "search phase, returning.\n");
            }
            free(cur_costs);
            free(sorted_prizes);
            free(sorted_indices);
            return true;
        }
        if (cur_k > sparsity_high) {
            lambda_low = lambda_mid;
        } else {
            lambda_high = lambda_mid;
        }
    }
    for (int ii = 0; ii < m; ++ii) {
        cur_costs[ii] = lambda_high * costs[ii];
    }
    PCST *pcst = make_pcst(edges, prizes, cur_costs, root, target_num_clusters,
                           1e-10, pruning, n, m, verbose);
    run_pcst(pcst, stat->re_nodes, stat->re_edges);
    free_pcst(pcst);
    if (verbose >= 1) {
        for (int ii = 0; ii < m; ii++) {
            printf("E %d %d %.15f\n", edges[ii].first, edges[ii].second,
                   cur_costs[ii]);
        }
        printf("\n");
        for (int ii = 0; ii < n; ii++) {
            printf("N %d %.15f\n", ii, prizes[ii]);
        }
        printf("\n");
        printf("Reached the maximum number of "
               "iterations, using the last l_high: %e  k: %d\n",
               lambda_high, stat->re_nodes->size);
    }
    free(cur_costs);
    free(sorted_prizes);
    free(sorted_indices);
    return true;
}


/**
 * calculate the TPR, FPR, and AUC score.
 *
 */

void _tpr_fpr_auc(const double *true_labels,
                  const double *scores, int n, double *tpr, double *fpr, double *auc) {
    double num_posi = 0.0;
    double num_nega = 0.0;
    for (int i = 0; i < n; i++) {
        if (true_labels[i] > 0) {
            num_posi++;
        } else {
            num_nega++;
        }
    }
    int *sorted_indices = malloc(sizeof(int) * n);
    _arg_sort_descend(scores, sorted_indices, n);
    //TODO assume the score has no -infty
    tpr[0] = 0.0; // initial point.
    fpr[0] = 0.0; // initial point.
    // accumulate sum
    for (int i = 0; i < n; i++) {
        double cur_label = true_labels[sorted_indices[i]];
        if (cur_label > 0) {
            fpr[i + 1] = fpr[i];
            tpr[i + 1] = tpr[i] + 1.0;
        } else {
            fpr[i + 1] = fpr[i] + 1.0;
            tpr[i + 1] = tpr[i];
        }
    }
    cblas_dscal(n, 1. / num_posi, tpr, 1);
    cblas_dscal(n, 1. / num_nega, fpr, 1);
    //AUC score by
    *auc = 0.0;
    double prev = 0.0;
    for (int i = 0; i < n; i++) {
        *auc += (tpr[i] * (fpr[i] - prev));
        prev = fpr[i];
    }
    free(sorted_indices);
}

/**
 * Calculate the AUC score.
 * We assume true labels contain only +1,-1
 * We also assume scores are real numbers.
 * @param true_labels
 * @param scores
 * @param len
 * @return AUC score.
 */
double _auc_score(const double *true_labels, const double *scores, int len) {
    double *fpr = malloc(sizeof(double) * (len + 1));
    double *tpr = malloc(sizeof(double) * (len + 1));
    double num_posi = 0.0;
    double num_nega = 0.0;
    for (int i = 0; i < len; i++) {
        if (true_labels[i] > 0) {
            num_posi++;
        } else {
            num_nega++;
        }
    }
    int *sorted_indices = malloc(sizeof(int) * len);
    _arg_sort_descend(scores, sorted_indices, len);
    tpr[0] = 0.0; // initial point.
    fpr[0] = 0.0; // initial point.
    // accumulate sum
    for (int i = 0; i < len; i++) {
        double cur_label = true_labels[sorted_indices[i]];
        if (cur_label > 0) {
            fpr[i + 1] = fpr[i];
            tpr[i + 1] = tpr[i] + 1.0;
        } else {
            fpr[i + 1] = fpr[i] + 1.0;
            tpr[i + 1] = tpr[i];
        }
    }
    cblas_dscal(len, 1. / num_posi, tpr, 1);
    cblas_dscal(len, 1. / num_nega, fpr, 1);
    //AUC score
    double auc = 0.0;
    double prev = 0.0;
    for (int i = 0; i < len; i++) {
        auc += (tpr[i] * (fpr[i] - prev));
        prev = fpr[i];
    }
    free(sorted_indices);
    free(fpr);
    free(tpr);
    return auc;
}

void _sparse_to_full(const double *sparse_v, const int *sparse_indices,
                     int sparse_len, double *full_v, int full_len) {
    cblas_dscal(full_len, 0.0, full_v, 1);
    for (int i = 0; i < sparse_len; i++) {
        full_v[sparse_indices[i]] = sparse_v[i];
    }
}

/**
 * Please find the algorithm in the following paper:
 * ---
 * @article{floyd1975algorithm,
 * title={Algorithm 489: the algorithm SELECT—for finding the ith
 *        smallest of n elements [M1]},
 * author={Floyd, Robert W and Rivest, Ronald L},
 * journal={Communications of the ACM},
 * volume={18}, number={3}, pages={173},
 * year={1975},
 * publisher={ACM}}
 * @param array
 * @param l
 * @param r
 * @param k
 */
void _floyd_rivest_select(double *array, int l, int r, int k) {
    register int n, i, j, s, sd, ll, rr;
    register double z, t;
    while (r > l) {
        if (r - l > 600) {
            /**
             * use select() recursively on a sample of size s to get an
             * estimate for the (k-l+1)-th smallest element into array[k],
             * biased slightly so that the (k-l+1)-th element is expected to
             * lie in the smaller set after partitioning.
             */
            n = r - l + 1;
            i = k - l + 1;
            z = log(n);
            s = (int) (0.5 * exp(2 * z / 3));
            sd = (int) (0.5 * sqrt(z * s * (n - s) / n) * sign(i - n / 2));
            ll = max(l, k - i * s / n + sd);
            rr = min(r, k + (n - i) * s / n + sd);
            _floyd_rivest_select(array, ll, rr, k);
        }
        t = array[k];
        /**
         * the following code partitions x[l:r] about t, it is similar to partition
         * but will run faster on most machines since subscript range checking on i
         * and j has been eliminated.
         */
        i = l;
        j = r;
        swap(array[l], array[k]);
        if (array[r] < t) {
            swap(array[r], array[l]);
        }
        while (i < j) {
            swap(array[i], array[j]);
            do i++; while (array[i] > t);
            do j--; while (array[j] < t);
        }
        if (array[l] == t) {
            swap(array[l], array[j]);
        } else {
            j++;
            swap(array[j], array[r]);
        }
        /**
         * New adjust l, r so they surround the subset containing the
         * (k-l+1)-th smallest element.
         */
        if (j <= k) {
            l = j + 1;
        }
        if (k <= j) {
            r = j - 1;
        }
    }
}


/**
 * Given the unsorted array, we threshold this array by using Floyd-Rivest algorithm.
 * @param arr the unsorted array.
 * @param n, the number of elements in this array.
 * @param k, the number of k largest elements will be kept.
 * @return 0, successfully project arr to a k-sparse vector.
 */
int _hard_thresholding(double *arr, int n, int k) {
    double *temp_arr = malloc(sizeof(double) * n), kth_largest;
    for (int i = 0; i < n; i++) {
        temp_arr[i] = fabs(arr[i]);
    }
    _floyd_rivest_select(temp_arr, 0, n - 1, k - 1);
    kth_largest = temp_arr[k - 1];
    bool flag = false;
    for (int i = 0; i < n; i++) {
        if (fabs(arr[i]) < kth_largest) {
            arr[i] = 0.0;
        } else if ((fabs(arr[i]) == kth_largest) && (flag == false)) {
            flag = true; // to handle the multiple cases.
        } else if ((fabs(arr[i]) == kth_largest) && (flag == true)) {
            arr[i] = 0.0;
        }
    }
    free(temp_arr);
    return 0;
}

/**
 * This code is implemented by Laurent Condat, PhD, CNRS research fellow in France.
 * @param y
 * @param x
 * @param length
 * @param a
 */
static void _l1ballproj_condat(double *y, double *x, int length, const double a) {
    if (a <= 0.0) {
        if (a == 0.0) memset(x, 0, length * sizeof(double));
        return;
    }
    double *aux = (x == y ? (double *) malloc(length * sizeof(double)) : x);
    int aux_len = 1;
    int aux_len_hold = -1;
    double tau = (*aux = (*y >= 0.0 ? *y : -*y)) - a;
    int i = 1;
    for (; i < length; i++) {
        if (y[i] > 0.0) {
            if (y[i] > tau) {
                if ((tau += ((aux[aux_len] = y[i]) - tau) / (aux_len - aux_len_hold)) <=
                    y[i] - a) {
                    tau = y[i] - a;
                    aux_len_hold = aux_len - 1;
                }
                aux_len++;
            }
        } else if (y[i] != 0.0) {
            if (-y[i] > tau) {
                if ((tau += ((aux[aux_len] = -y[i]) - tau) / (aux_len - aux_len_hold))
                    <= aux[aux_len] - a) {
                    tau = aux[aux_len] - a;
                    aux_len_hold = aux_len - 1;
                }
                aux_len++;
            }
        }
    }
    if (tau <= 0) {    /* y is in the l1 ball => x=y */
        if (x != y) memcpy(x, y, length * sizeof(double));
        else free(aux);
    } else {
        double *aux0 = aux;
        if (aux_len_hold >= 0) {
            aux_len -= ++aux_len_hold;
            aux += aux_len_hold;
            while (--aux_len_hold >= 0)
                if (aux0[aux_len_hold] > tau)
                    tau += ((*(--aux) = aux0[aux_len_hold]) - tau) / (++aux_len);
        }
        do {
            aux_len_hold = aux_len - 1;
            for (i = aux_len = 0; i <= aux_len_hold; i++)
                if (aux[i] > tau)
                    aux[aux_len++] = aux[i];
                else
                    tau += (tau - aux[i]) / (aux_len_hold - i + aux_len);
        } while (aux_len <= aux_len_hold);
        for (i = 0; i < length; i++)
            x[i] = (y[i] - tau > 0.0 ? y[i] - tau : (y[i] + tau < 0.0 ? y[i] + tau : 0.0));
        if (x == y) free(aux0);
    }
}


bool _algo_solam(const double *data_x_tr,
                 const double *data_y_tr,
                 int data_n,
                 int data_p,
                 double para_xi,
                 double para_r,
                 int para_num_pass,
                 int para_step_len,
                 int para_verbose,
                 double *re_wt,
                 double *re_wt_bar,
                 double *re_auc,
                 double *re_rts) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);
    double start_time = clock();
    if (para_verbose > 0) { printf("n: %d p: %d", data_n, data_p); }
    double gamma, t = 1.0, p_hat = 0.; // learning rate, iteration time, positive ratio.
    double gamma_bar, gamma_bar_prev = 0.0, alpha_bar, alpha_bar_prev = 0.0;
    double *v = malloc(sizeof(double) * (data_p + 2));
    // initialize vt
    double *v_prev = malloc(sizeof(double) * (data_p + 2));
    for (int i = 0; i < data_p; i++) {
        v_prev[i] = sqrt(para_r * para_r / (double) data_p);
    }
    v_prev[data_p] = para_r;
    v_prev[data_p + 1] = para_r;
    // initialize alpha1
    double alpha, alpha_prev = 2. * para_r;
    double *grad_v = malloc(sizeof(double) * (data_p + 2));
    double *v_bar = malloc(sizeof(double) * (data_p + 2));
    double *v_bar_prev = malloc(sizeof(double) * (data_p + 2));
    memset(v_bar_prev, 0, sizeof(double) * (data_p + 2));
    cblas_dcopy(data_p, v_bar_prev, 1, re_wt, 1);
    cblas_dcopy(data_p, v_bar_prev, 1, re_wt_bar, 1);
    double *y_pred = malloc(sizeof(double) * data_n);
    int auc_index = 0;
    for (int i = 0; i < para_num_pass; i++) {
        for (int j = 0; j < data_n; j++) {
            const double *xt = data_x_tr + j * data_p; // current sample
            double is_p_yt = is_posi(data_y_tr[j]);
            double is_n_yt = is_nega(data_y_tr[j]);
            p_hat = ((t - 1.) * p_hat + is_p_yt) / t; // update p_hat
            gamma = para_xi / sqrt(t); // current learning rate
            cblas_dcopy(data_p, xt, 1, grad_v, 1); // calculate the gradient w
            double vt_dot = cblas_ddot(data_p, v_prev, 1, xt, 1);
            double wei_posi = 2. * (1. - p_hat) * (vt_dot - v_prev[data_p] - (1. + alpha_prev));
            double wei_nega = 2. * p_hat * ((vt_dot - v_prev[data_p + 1]) + (1. + alpha_prev));
            double weight = wei_posi * is_p_yt + wei_nega * is_n_yt;
            cblas_dscal(data_p, weight, grad_v, 1);
            grad_v[data_p] = -2. * (1. - p_hat) * (vt_dot - v_prev[data_p]) * is_p_yt; //grad a
            grad_v[data_p + 1] = -2. * p_hat * (vt_dot - v_prev[data_p + 1]) * is_n_yt; //grad b
            cblas_dscal(data_p + 2, -gamma, grad_v, 1); // gradient descent step of vt
            cblas_daxpy(data_p + 2, 1.0, v_prev, 1, grad_v, 1);
            cblas_dcopy(data_p + 2, grad_v, 1, v, 1);
            wei_posi = -2. * (1. - p_hat) * vt_dot; // calculate the gradient of dual alpha
            wei_nega = 2. * p_hat * vt_dot;
            double grad_alpha = wei_posi * is_p_yt + wei_nega * is_n_yt;
            grad_alpha += -2. * p_hat * (1. - p_hat) * alpha_prev;
            alpha = alpha_prev + gamma * grad_alpha; // gradient descent step of alpha
            double norm_v = sqrt(cblas_ddot(data_p, v, 1, v, 1)); // projection w
            if (norm_v > para_r) { cblas_dscal(data_p, para_r / norm_v, v, 1); }
            v[data_p] = (v[data_p] > para_r) ? para_r : v[data_p]; // projection a
            v[data_p + 1] = (v[data_p + 1] > para_r) ? para_r : v[data_p + 1]; // projection b
            // projection alpha
            alpha = (fabs(alpha) > 2. * para_r) ? (2. * alpha * para_r) / fabs(alpha) : alpha;
            gamma_bar = gamma_bar_prev + gamma; // update gamma_
            cblas_dcopy(data_p + 2, v_prev, 1, v_bar, 1); // update v_bar
            cblas_dscal(data_p + 2, gamma / gamma_bar, v_bar, 1);
            cblas_daxpy(data_p + 2, gamma_bar_prev / gamma_bar, v_bar_prev, 1, v_bar, 1);
            // update alpha_bar
            alpha_bar = (gamma_bar_prev * alpha_bar_prev + gamma * alpha_prev) / gamma_bar;
            cblas_dscal(data_p, (t - 1.) / t, re_wt_bar, 1);
            cblas_daxpy(data_p, 1. / t, v_bar, 1, re_wt_bar, 1);
            // update the parameters.
            alpha_prev = alpha;
            alpha_bar_prev = alpha_bar;
            gamma_bar_prev = gamma_bar;
            cblas_dcopy(data_p + 2, v_bar, 1, v_bar_prev, 1);
            cblas_dcopy(data_p + 2, v, 1, v_prev, 1);
            if ((fmod(t, para_step_len) == 0.)) { // to calculate AUC score
                double t_eval = clock();
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
                re_auc[auc_index] = _auc_score(data_y_tr, y_pred, data_n);
                t_eval = clock() - t_eval;
                re_rts[auc_index++] = (clock() - start_time - t_eval) / CLOCKS_PER_SEC;
            }
            t = t + 1.; // update the counts
        }
    }
    cblas_dcopy(data_p, v_bar, 1, re_wt, 1);
    free(y_pred);
    free(v_bar_prev);
    free(v_bar);
    free(grad_v);
    free(v_prev);
    free(v);
    return true;
}

bool _algo_solam_sparse(const double *x_tr_vals,
                        const int *x_tr_indices,
                        const int *x_tr_posis,
                        const int *x_tr_lens,
                        const double *data_y_tr,
                        int data_n,
                        int data_p,
                        double para_xi,
                        double para_r,
                        int para_num_passes,
                        int para_step_len,
                        int para_verbose,
                        double *re_wt,
                        double *re_wt_bar,
                        double *re_auc,
                        double *re_rts) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);
    double start_time = clock();
    if (para_verbose > 0) { printf("n: %d p: %d", data_n, data_p); }
    double gamma, t = 1.0, p_hat = 0.; // learning rate, iteration time, positive ratio.
    double gamma_bar, gamma_bar_prev = 0.0, alpha_bar, alpha_bar_prev = 0.0;
    double *v = malloc(sizeof(double) * (data_p + 2));
    double *v_prev = malloc(sizeof(double) * (data_p + 2)); // initialize vt
    for (int i = 0; i < data_p; i++) {
        v_prev[i] = sqrt(para_r * para_r / (double) data_p);
    }
    v_prev[data_p] = para_r;
    v_prev[data_p + 1] = para_r;
    double alpha, alpha_prev = 2. * para_r; // initialize alpha1
    double *grad_v = malloc(sizeof(double) * (data_p + 2));
    double *v_bar = malloc(sizeof(double) * (data_p + 2));
    double *v_bar_prev = malloc(sizeof(double) * (data_p + 2));
    double *xt = malloc(sizeof(double) * (data_p));
    double *y_pred = malloc(sizeof(double) * (data_n));
    memset(v_bar_prev, 0, sizeof(double) * (data_p + 2));
    cblas_dcopy(data_p, v_bar_prev, 1, re_wt, 1);
    cblas_dcopy(data_p, v_bar_prev, 1, re_wt_bar, 1);
    int auc_index = 0;
    for (int i = 0; i < para_num_passes; i++) {
        for (int j = 0; j < data_n; j++) {
            const int *xt_indices = x_tr_indices + x_tr_posis[j]; // current sample
            const double *xt_vals = x_tr_vals + x_tr_posis[j];
            memset(xt, 0, sizeof(double) * data_p);
            for (int kk = 0; kk < x_tr_lens[j]; kk++) {
                xt[xt_indices[kk]] = xt_vals[kk];
            }
            double is_p_yt = is_posi(data_y_tr[j]);
            double is_n_yt = is_nega(data_y_tr[j]);
            p_hat = ((t - 1.) * p_hat + is_p_yt) / t; // update p_hat
            gamma = para_xi / sqrt(t); // current learning rate
            cblas_dcopy(data_p, xt, 1, grad_v, 1); // calculate the gradient w
            double vt_dot = cblas_ddot(data_p, v_prev, 1, xt, 1);
            double wei_posi = 2. * (1. - p_hat) * (vt_dot - v_prev[data_p] - (1. + alpha_prev));
            double wei_nega = 2. * p_hat * ((vt_dot - v_prev[data_p + 1]) + (1. + alpha_prev));
            double weight = wei_posi * is_p_yt + wei_nega * is_n_yt;
            cblas_dscal(data_p, weight, grad_v, 1);
            grad_v[data_p] = -2. * (1. - p_hat) * (vt_dot - v_prev[data_p]) * is_p_yt; //grad of a
            grad_v[data_p + 1] = -2. * p_hat * (vt_dot - v_prev[data_p + 1]) * is_n_yt; //grad of b
            cblas_dscal(data_p + 2, -gamma, grad_v, 1); // gradient descent step of vt
            cblas_daxpy(data_p + 2, 1.0, v_prev, 1, grad_v, 1);
            cblas_dcopy(data_p + 2, grad_v, 1, v, 1);
            wei_posi = -2. * (1. - p_hat) * vt_dot; // calculate the gradient of dual alpha
            wei_nega = 2. * p_hat * vt_dot;
            double grad_alpha = wei_posi * is_p_yt + wei_nega * is_n_yt;
            grad_alpha += -2. * p_hat * (1. - p_hat) * alpha_prev;
            alpha = alpha_prev + gamma * grad_alpha; // gradient descent step of alpha
            double norm_v = sqrt(cblas_ddot(data_p, v, 1, v, 1)); // projection w
            if (norm_v > para_r) { cblas_dscal(data_p, para_r / norm_v, v, 1); }
            v[data_p] = (v[data_p] > para_r) ? para_r : v[data_p]; // projection a,b
            v[data_p + 1] = (v[data_p + 1] > para_r) ? para_r : v[data_p + 1]; // projection alpha
            alpha = (fabs(alpha) > 2. * para_r) ? (2. * alpha * para_r) / fabs(alpha) : alpha;
            gamma_bar = gamma_bar_prev + gamma; // update gamma_
            cblas_dcopy(data_p + 2, v_prev, 1, v_bar, 1); // update v_bar
            cblas_dscal(data_p + 2, gamma / gamma_bar, v_bar, 1);
            cblas_daxpy(data_p + 2, gamma_bar_prev / gamma_bar, v_bar_prev, 1, v_bar, 1);
            // update alpha_bar
            alpha_bar = (gamma_bar_prev * alpha_bar_prev + gamma * alpha_prev) / gamma_bar;
            cblas_dscal(data_p, (t - 1.) / t, re_wt_bar, 1);
            cblas_daxpy(data_p, 1. / t, v_bar, 1, re_wt_bar, 1);
            alpha_prev = alpha; // update the parameters.
            alpha_bar_prev = alpha_bar;
            gamma_bar_prev = gamma_bar;
            cblas_dcopy(data_p + 2, v_bar, 1, v_bar_prev, 1);
            cblas_dcopy(data_p + 2, v, 1, v_prev, 1);
            if ((fmod(t, para_step_len) == 1.)) { // to calculate AUC score
                double cur_t_s = clock();
                for (int q = 0; q < data_n; q++) {
                    memset(xt, 0, sizeof(double) * data_p);
                    xt_indices = x_tr_indices + x_tr_posis[q];
                    xt_vals = x_tr_vals + x_tr_posis[q];
                    for (int tt = 0; tt < x_tr_lens[q]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
                    y_pred[q] = cblas_ddot(data_p, xt, 1, v_bar, 1);
                }
                re_auc[auc_index] = _auc_score(data_y_tr, y_pred, data_n);
                double eval_time = clock() - cur_t_s;
                re_rts[auc_index++] = (clock() - start_time - eval_time) / CLOCKS_PER_SEC;
            }
            // update the counts
            t = t + 1.;
        }
    }
    cblas_dcopy(data_p, v_bar, 1, re_wt, 1);
    free(y_pred);
    free(xt);
    free(v_bar_prev);
    free(v_bar);
    free(grad_v);
    free(v_prev);
    free(v);
    return true;
}

void _algo_spam(const double *data_x_tr,
                const double *data_y_tr,
                int data_n,
                int data_p,
                double para_xi,
                double para_l1_reg,
                double para_l2_reg,
                int para_num_passes,
                int para_step_len,
                int para_reg_opt,
                int para_verbose,
                double *re_wt,
                double *re_wt_bar,
                double *re_auc) {

    openblas_set_num_threads(1); // make sure openblas uses only one cpu at a time.
    memset(re_wt, 0, sizeof(double) * data_p); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * data_p); // wt_bar --> 0.0
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    double *u = malloc(sizeof(double) * data_p); // proxy vector
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * data_p);
    memset(posi_x_mean, 0, sizeof(double) * data_p);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * data_p);
    memset(nega_x_mean, 0, sizeof(double) * data_p);
    double alpha_wt; // initialize alpha_wt (initialize to zero.)
    // to determine a_wt, b_wt, and alpha_wt based on all training samples.
    double posi_t = 0.0, nega_t = 0.0;
    double *y_pred = malloc(sizeof(double) * data_n);
    for (int i = 0; i < data_n; i++) {
        if (data_y_tr[i] > 0) {
            posi_t++;
            cblas_dscal(data_p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(data_p, 1. / posi_t, (data_x_tr + i * data_p), 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(data_p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(data_p, 1. / nega_t, (data_x_tr + i * data_p), 1, nega_x_mean, 1);
        }
    }
    // the estimate of Pr(y=1), learning rate, and initialize start time.
    double prob_p = posi_t / (data_n * 1.0), eta_t, t = 1.0;
    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
    }
    int auc_index = 0;
    for (int i = 0; i < para_num_passes; i++) { // for each epoch
        for (int j = 0; j < data_n; j++) { // for each training sample
            eta_t = para_xi / sqrt(t); // current learning rate
            a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1); // update a(wt)
            b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1); // para_b(wt)
            alpha_wt = b_wt - a_wt; // alpha(wt)
            double wt_dot = cblas_ddot(data_p, re_wt, 1, (data_x_tr + j * data_p), 1);
            double weight = data_y_tr[j] > 0 ? 2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                                               2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                            2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
            cblas_dcopy(data_p, (data_x_tr + j * data_p), 1, grad_wt, 1); //calculate the gradient
            cblas_dscal(data_p, weight, grad_wt, 1);
            cblas_dcopy(data_p, re_wt, 1, u, 1); // gradient descent: u= wt - eta * grad(wt)
            cblas_daxpy(data_p, -eta_t, grad_wt, 1, u, 1);
            /**
             * Currently, u is the \hat{wt_{t+1}}, next is to use prox_operator.
             * The following part of the code is the proximal operator for
             * 1. para_reg_opt==0: elastic regularization.
             * 2. para_reg_opt==1: ell_2 regularization.
             * ---
             * @inproceedings{singer2009efficient,
             * title={Efficient learning using forward-backward splitting},
             * author={Singer, Yoram and Duchi, John C},
             * booktitle={Advances in Neural Information Processing Systems},
             * pages={495--503},
             * year={2009}}
             */
            if (para_reg_opt == 0) { // elastic-net
                double tmp_l2 = (eta_t * para_l2_reg + 1.);
                for (int k = 0; k < data_p; k++) {
                    double sign_uk = (double) (sign(u[k]));
                    re_wt[k] = (sign_uk / tmp_l2) * fmax(0.0, fabs(u[k]) - eta_t * para_l1_reg);
                }
            } else { // l2-regularization
                cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
                cblas_dcopy(data_p, u, 1, re_wt, 1);
            }
            cblas_dscal(data_p, (t - 1.) / t, re_wt_bar, 1); // take average of wt
            cblas_daxpy(data_p, 1. / t, re_wt, 1, re_wt_bar, 1);
            if ((fmod(t, para_step_len) == 0.)) { // to calculate AUC score
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
                re_auc[auc_index++] = _auc_score(data_y_tr, y_pred, data_n);
            }
            t = t + 1.0; // increase time
        }
    }
    free(y_pred);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
}

void _algo_spam_sparse(const double *x_tr_vals,
                       const int *x_tr_indices,
                       const int *x_tr_posis,
                       const int *x_tr_lens,
                       const double *data_y_tr,
                       int data_n,
                       int data_p,
                       double para_xi,
                       double para_l1_reg,
                       double para_l2_reg,
                       int para_num_passes,
                       int para_step_len,
                       int para_reg_opt,
                       int para_verbose,
                       double *re_wt,
                       double *re_wt_bar,
                       double *re_auc,
                       double *re_rts) {

    openblas_set_num_threads(1); // make sure openblas uses only one cpu at a time.
    double start_time = clock();
    memset(re_wt, 0, sizeof(double) * data_p); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * data_p); // wt_bar --> 0.0
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    double *u = malloc(sizeof(double) * data_p); // proxy vector
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * data_p);
    memset(posi_x_mean, 0, sizeof(double) * data_p);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * data_p);
    memset(nega_x_mean, 0, sizeof(double) * data_p);
    double alpha_wt; // initialize alpha_wt (initialize to zero.)
    // to determine a_wt, b_wt, and alpha_wt based on all training samples.
    double posi_t = 0.0, nega_t = 0.0;
    double *y_pred = malloc(sizeof(double) * data_n);
    double *xt = malloc(sizeof(double) * data_p);
    for (int i = 0; i < data_n; i++) {
        // receive training sample zt=(xt,yt)
        const int *xt_indices = x_tr_indices + x_tr_posis[i];
        const double *xt_vals = x_tr_vals + x_tr_posis[i];
        memset(xt, 0, sizeof(double) * data_p);
        for (int tt = 0; tt < x_tr_lens[i]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
        if (data_y_tr[i] > 0) {
            posi_t++;
            cblas_dscal(data_p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(data_p, 1. / posi_t, xt, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(data_p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(data_p, 1. / nega_t, xt, 1, nega_x_mean, 1);
        }
    }
    // the estimate of Pr(y=1), learning rate, and initialize start time.
    double prob_p = posi_t / (data_n * 1.0), eta_t, t = 1.0;
    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
    }
    int auc_index = 0;
    for (int i = 0; i < para_num_passes; i++) { // for each epoch
        for (int j = 0; j < data_n; j++) { // for each training sample
            // receive training sample zt=(xt,yt)
            const int *xt_indices = x_tr_indices + x_tr_posis[j];
            const double *xt_vals = x_tr_vals + x_tr_posis[j];
            memset(xt, 0, sizeof(double) * data_p);
            for (int tt = 0; tt < x_tr_lens[j]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
            eta_t = para_xi / sqrt(t); // current learning rate
            a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1); // update a(wt)
            b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1); // para_b(wt)
            alpha_wt = b_wt - a_wt; // alpha(wt)
            double wt_dot = cblas_ddot(data_p, re_wt, 1, xt, 1);
            double weight = data_y_tr[j] > 0 ? 2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                                               2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                            2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
            cblas_dcopy(data_p, xt, 1, grad_wt, 1); // calculate the gradient
            cblas_dscal(data_p, weight, grad_wt, 1);
            cblas_dcopy(data_p, re_wt, 1, u, 1); // gradient descent: u= wt - eta * grad(wt)
            cblas_daxpy(data_p, -eta_t, grad_wt, 1, u, 1);
            /**
             * Currently, u is the \hat{wt_{t+1}}, next is to use prox_operator.
             * The following part of the code is the proximal operator for
             * 1. para_reg_opt==0: elastic regularization.
             * 2. para_reg_opt==1: ell_2 regularization.
             * ---
             * @inproceedings{singer2009efficient,
             * title={Efficient learning using forward-backward splitting},
             * author={Singer, Yoram and Duchi, John C},
             * booktitle={Advances in Neural Information Processing Systems},
             * pages={495--503},
             * year={2009}}
             */
            if (para_reg_opt == 0) { // elastic-net
                double tmp_l2 = (eta_t * para_l2_reg + 1.);
                for (int k = 0; k < data_p; k++) {
                    double sign_uk = (double) (sign(u[k]));
                    re_wt[k] = (sign_uk / tmp_l2) * fmax(0.0, fabs(u[k]) - eta_t * para_l1_reg);
                }
            } else { // l2-regularization
                cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
                cblas_dcopy(data_p, u, 1, re_wt, 1);
            }
            cblas_dscal(data_p, (t - 1.) / t, re_wt_bar, 1); // take average of wt
            cblas_daxpy(data_p, 1. / t, re_wt, 1, re_wt_bar, 1);
            if ((fmod(t, para_step_len) == 1.)) { // to calculate AUC score
                double cur_t_s = clock();
                for (int q = 0; q < data_n; q++) {
                    memset(xt, 0, sizeof(double) * data_p);
                    xt_indices = x_tr_indices + x_tr_posis[q];
                    xt_vals = x_tr_vals + x_tr_posis[q];
                    for (int tt = 0; tt < x_tr_lens[q]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
                    y_pred[q] = cblas_ddot(data_p, xt, 1, re_wt, 1);
                }
                re_auc[auc_index] = _auc_score(data_y_tr, y_pred, data_n);
                double eval_time = clock() - cur_t_s;
                re_rts[auc_index++] = (clock() - start_time - eval_time) / CLOCKS_PER_SEC;
            }
            t = t + 1.0; // increase time
        }
    }
    free(xt);
    free(y_pred);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
}


void _algo_sht_am(const double *data_x_tr,
                  const double *data_y_tr,
                  int data_n,
                  int data_p,
                  int para_sparsity,
                  int para_b,
                  double para_xi,
                  double para_l2_reg,
                  int para_num_passes,
                  int para_step_len,
                  int para_verbose,
                  double *re_wt,
                  double *re_wt_bar,
                  double *re_auc) {

    openblas_set_num_threads(1); // make sure openblas uses only one cpu at a time.
    memset(re_wt, 0, sizeof(double) * data_p); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * data_p); // wt_bar --> 0.0
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    double *u = malloc(sizeof(double) * data_p); // proxy vector
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * data_p);
    memset(posi_x_mean, 0, sizeof(double) * data_p); // posi_x_mean --> 0.0
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * data_p);
    memset(nega_x_mean, 0, sizeof(double) * data_p); // nega_x_mean --> 0.0
    double alpha_wt; // initialize alpha_wt (initialize to zero.)
    double posi_t = 0.0, nega_t = 0.0; // to determine a_wt, b_wt, and alpha_wt
    for (int i = 0; i < data_n; i++) {
        if (data_y_tr[i] > 0) {
            posi_t++;
            cblas_dscal(data_p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(data_p, 1. / posi_t, (data_x_tr + i * data_p), 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(data_p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(data_p, 1. / nega_t, (data_x_tr + i * data_p), 1, nega_x_mean, 1);
        }
    }
    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (data_n * 1.0), eta_t, t = 1.0;
    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
    }
    double *aver_grad = malloc(sizeof(double) * data_p);
    memset(aver_grad, 0, sizeof(double) * data_p);
    double *y_pred = malloc(sizeof(double) * data_n);
    int auc_index = 0;
    for (int i = 0; i < para_num_passes; i++) { // for each epoch
        for (int j = 0; j < data_n / para_b; j++) { // n/b is the total number of blocks.
            // update a(wt), para_b(wt), and alpha(wt)
            a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;
            eta_t = para_xi / sqrt(t); // current learning rate
            // receive a block of training samples to calculate the gradient
            memset(grad_wt, 0, sizeof(double) * data_p);
            for (int kk = 0; kk < para_b; kk++) { // for each training sample j
                const double *cur_xt = data_x_tr + j * para_b * data_p + kk * data_p;
                double cur_yt = data_y_tr[j * para_b + kk];
                double wt_dot = cblas_ddot(data_p, re_wt, 1, cur_xt, 1);
                double weight = cur_yt > 0 ? 2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                                             2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                                2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
                cblas_daxpy(data_p, weight, cur_xt, 1, grad_wt, 1); // calculate the gradient
            }
            cblas_dscal(data_p, (t - 1.) / t, aver_grad, 1); // take average of wt --> wt_bar
            cblas_daxpy(data_p, 1. / t, grad_wt, 1, aver_grad, 1);
            cblas_dscal(data_p, 1. / (para_b * 1.0), grad_wt, 1);
            cblas_dcopy(data_p, re_wt, 1, u, 1); //gradient descent: u= wt - eta * grad(wt)
            cblas_daxpy(data_p, -eta_t, grad_wt, 1, u, 1);
            /**
             * ell_2 regularization option proposed in the following paper:
             * @inproceedings{singer2009efficient,
             * title={Efficient learning using forward-backward splitting},
             * author={Singer, Yoram and Duchi, John C},
             * booktitle={Advances in Neural Information Processing Systems},
             * pages={495--503},
             * year={2009}}
             */
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
            _hard_thresholding(u, data_p, para_sparsity); // k-sparse step.
            cblas_dcopy(data_p, u, 1, re_wt, 1);
            cblas_dscal(data_p, (t - 1.) / t, re_wt_bar, 1);
            cblas_daxpy(data_p, 1. / t, re_wt, 1, re_wt_bar, 1);
            // to calculate AUC score and run time
            if ((fmod(t, para_step_len) == 0.)) {
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
                double auc = _auc_score(data_y_tr, y_pred, data_n);
                re_auc[auc_index++] = auc;
            }
            t = t + 1.0; // increase time
        }
    }
    free(y_pred);
    free(aver_grad);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
}


void _algo_sht_am_sparse(const double *x_tr_vals,
                         const int *x_tr_indices,
                         const int *x_tr_posis,
                         const int *x_tr_lens,
                         const double *data_y_tr,
                         int data_n,
                         int data_p,
                         int para_sparsity,
                         int para_b,
                         double para_xi,
                         double para_l2_reg,
                         int para_num_passes,
                         int para_step_len,
                         int para_verbose,
                         double *re_wt,
                         double *re_wt_bar,
                         double *re_auc,
                         double *re_rts) {

    openblas_set_num_threads(1); // make sure openblas uses only one cpu at a time.
    double start_time = clock();
    memset(re_wt, 0, sizeof(double) * data_p); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * data_p); // wt_bar --> 0.0
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    double *u = malloc(sizeof(double) * data_p); // proxy vector
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * data_p);
    memset(posi_x_mean, 0, sizeof(double) * data_p); // posi_x_mean --> 0.0
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * data_p);
    memset(nega_x_mean, 0, sizeof(double) * data_p); // nega_x_mean --> 0.0
    double alpha_wt; // initialize alpha_wt (initialize to zero.)
    double *xt = malloc(sizeof(double) * data_p);
    double posi_t = 0.0, nega_t = 0.0; // to determine a_wt, b_wt, and alpha_wt
    for (int i = 0; i < data_n; i++) {
        const int *xt_indices = x_tr_indices + x_tr_posis[i];
        const double *xt_vals = x_tr_vals + x_tr_posis[i];
        memset(xt, 0, sizeof(double) * data_p);
        for (int kk = 0; kk < x_tr_lens[i]; kk++) { xt[xt_indices[kk]] = xt_vals[kk]; }
        if (data_y_tr[i] > 0) {
            posi_t++;
            cblas_dscal(data_p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(data_p, 1. / posi_t, xt, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(data_p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(data_p, 1. / nega_t, xt, 1, nega_x_mean, 1);
        }
    }
    // Pr(y=1), learning rate, time clock.
    double prob_p = posi_t / (data_n * 1.0), eta_t, t = 1.0;
    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
    }
    double *aver_grad = malloc(sizeof(double) * data_p);
    memset(aver_grad, 0, sizeof(double) * data_p);
    double *y_pred = malloc(sizeof(double) * data_n);
    int auc_index = 0;
    for (int i = 0; i < para_num_passes; i++) { // for each epoch
        for (int j = 0; j < data_n / para_b; j++) { // data_n/b is the total number of blocks.
            // update a(wt), para_b(wt), and alpha(wt)
            a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;
            eta_t = para_xi / sqrt(t); // current learning rate
            // receive a block of training samples to calculate the gradient
            memset(grad_wt, 0, sizeof(double) * data_p);
            for (int kk = 0; kk < para_b; kk++) {
                int ind = j * para_b + kk; // current ind:
                const int *xt_indices = x_tr_indices + x_tr_posis[ind];
                const double *xt_vals = x_tr_vals + x_tr_posis[ind];
                memset(xt, 0, sizeof(double) * data_p);
                for (int tt = 0; tt < x_tr_lens[ind]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
                double cur_yt = data_y_tr[ind];
                double wt_dot = cblas_ddot(data_p, re_wt, 1, xt, 1);
                double weight = cur_yt > 0 ? 2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                                             2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                                2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
                cblas_daxpy(data_p, weight, xt, 1, grad_wt, 1); // calculate the gradient
            }
            cblas_dscal(data_p, (t - 1.) / t, aver_grad, 1); // take average of wt --> wt_bar
            cblas_daxpy(data_p, 1. / t, grad_wt, 1, aver_grad, 1);
            cblas_dscal(data_p, 1. / (para_b * 1.0), grad_wt, 1);
            cblas_dcopy(data_p, re_wt, 1, u, 1); //gradient descent: u= wt - eta * grad(wt)
            cblas_daxpy(data_p, -eta_t, grad_wt, 1, u, 1);
            /**
             * ell_2 regularization option proposed in the following paper:
             *
             * @inproceedings{singer2009efficient,
             * title={Efficient learning using forward-backward splitting},
             * author={Singer, Yoram and Duchi, John C},
             * booktitle={Advances in Neural Information Processing Systems},
             * pages={495--503},
             * year={2009}}
             */
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
            _hard_thresholding(u, data_p, para_sparsity); // k-sparse step.
            cblas_dcopy(data_p, u, 1, re_wt, 1);
            cblas_dscal(data_p, (t - 1.) / t, re_wt_bar, 1);
            cblas_daxpy(data_p, 1. / t, re_wt, 1, re_wt_bar, 1);
            if (true) { // to calculate AUC score
                double cur_t_s = clock();
                for (int q = 0; q < data_n; q++) {
                    memset(xt, 0, sizeof(double) * data_p);
                    const int *xt_indices = x_tr_indices + x_tr_posis[q];
                    const double *xt_vals = x_tr_vals + x_tr_posis[q];
                    for (int tt = 0; tt < x_tr_lens[q]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
                    y_pred[q] = cblas_ddot(data_p, xt, 1, re_wt, 1);
                }
                re_auc[auc_index] = _auc_score(data_y_tr, y_pred, data_n);
                double eval_time = clock() - cur_t_s;
                re_rts[auc_index++] = (clock() - start_time - eval_time) / CLOCKS_PER_SEC;
            }

            t = t + 1.; // increase time
        }
    }
    free(xt);
    free(aver_grad);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(y_pred);
    free(grad_wt);
}


void _algo_graph_am(const double *data_x_tr,
                    const double *data_y_tr,
                    const EdgePair *edges,
                    const double *weights,
                    int data_m,
                    int data_n,
                    int data_p,
                    int para_sparsity,
                    int para_b,
                    double para_xi,
                    double para_l2_reg,
                    int para_num_passes,
                    int para_step_len,
                    int para_verbose,
                    double *re_wt,
                    double *re_wt_bar,
                    double *re_auc) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);
    memset(re_wt, 0, sizeof(double) * data_p); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * data_p); // wt_bar --> 0.0
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    double *u = malloc(sizeof(double) * data_p); // proxy vector
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * data_p);
    memset(posi_x_mean, 0, sizeof(double) * data_p);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * data_p);
    memset(nega_x_mean, 0, sizeof(double) * data_p);
    double alpha_wt; // initialize alpha_wt (initialize to zero.)
    double posi_t = 0.0, nega_t = 0.0; // to determine a_wt, b_wt, and alpha_wt
    for (int i = 0; i < data_n; i++) {
        if (data_y_tr[i] > 0) {
            posi_t++;
            cblas_dscal(data_p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(data_p, 1. / posi_t, (data_x_tr + i * data_p), 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(data_p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(data_p, 1. / nega_t, (data_x_tr + i * data_p), 1, nega_x_mean, 1);
        }
    }
    // Pr(y=1), learning rate, initial start time is zero=1.0
    double prob_p = posi_t / (data_n * 1.0), eta_t, t = 1.0;
    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
    }
    double *aver_grad = malloc(sizeof(double) * data_p);
    memset(aver_grad, 0, sizeof(double) * data_p);
    double *proj_prizes = malloc(sizeof(double) * data_p);   // projected prizes.
    double *proj_costs = malloc(sizeof(double) * data_m);    // projected costs.
    GraphStat *graph_stat = make_graph_stat(data_p, data_m);   // head projection paras
    double *y_pred = malloc(sizeof(double) * data_n);
    int index_auc = 0;
    for (int i = 0; i < para_num_passes; i++) {
        for (int j = 0; j < data_n / para_b; j++) { // data_n/para_b is the total number of blocks.

            a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1); // update a(wt)
            b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1); // update b(wt)
            alpha_wt = b_wt - a_wt; // update alpha(wt)
            eta_t = para_xi / sqrt(t); // current learning rate
            // receive a block of training samples to calculate the gradient
            memset(grad_wt, 0, sizeof(double) * data_p);
            for (int kk = 0; kk < para_b; kk++) {
                int ind = (j * para_b + kk);
                const double *cur_xt = data_x_tr + ind * data_p;
                double wt_dot = cblas_ddot(data_p, re_wt, 1, cur_xt, 1);
                double weight = data_y_tr[ind] > 0 ? 2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                                                     2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                                2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
                cblas_daxpy(data_p, weight, cur_xt, 1, grad_wt, 1); // calculate the gradient
            }
            cblas_dscal(data_p, (t - 1.) / t, aver_grad, 1); // take average of wt --> wt_bar
            cblas_daxpy(data_p, 1. / t, grad_wt, 1, aver_grad, 1);
            cblas_dscal(data_p, 1. / (para_b * 1.0), grad_wt, 1);
            cblas_dcopy(data_p, re_wt, 1, u, 1); // gradient descent: u= wt - eta * grad(wt)
            cblas_daxpy(data_p, -eta_t, grad_wt, 1, u, 1);
            /**
             * ell_2 regularization option proposed in the following paper:
             *
             * @inproceedings{singer2009efficient,
             * title={Efficient learning using forward-backward splitting},
             * author={Singer, Yoram and Duchi, John C},
             * booktitle={Advances in Neural Information Processing Systems},
             * pages={495--503},
             * year={2009}}
             */
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
            //to do graph projection.
            for (int kk = 0; kk < data_p; kk++) { proj_prizes[kk] = u[kk] * u[kk]; }
            int g = 1, sparsity_low = para_sparsity, sparsity_high = para_sparsity + 10;
            int tail_max_iter = 20, verbose = 0;
            head_tail_binsearch(edges, weights, proj_prizes, data_p, data_m, g, -1, sparsity_low,
                                sparsity_high, tail_max_iter, GWPruning, verbose, graph_stat);
            cblas_dscal(data_p, 0.0, re_wt, 1);
            for (int kk = 0; kk < graph_stat->re_nodes->size; kk++) {
                int cur_node = graph_stat->re_nodes->array[kk];
                re_wt[cur_node] = u[cur_node];
            }
            // take average of wt --> wt_bar
            cblas_dscal(data_p, (t - 1.) / t, re_wt_bar, 1);
            cblas_daxpy(data_p, 1. / t, re_wt, 1, re_wt_bar, 1);
            // to calculate AUC score and run time
            if ((fmod(t, para_step_len) == 0.)) {
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
                re_auc[index_auc++] = _auc_score(data_y_tr, y_pred, data_n);
            }
            t = t + 1.0; // increase time
        }
    }
    free(y_pred);
    free(proj_prizes);
    free(proj_costs);
    free_graph_stat(graph_stat);
    free(aver_grad);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
}


void _algo_graph_am_sparse(const double *x_tr_vals,
                           const int *x_tr_indices,
                           const int *x_tr_posis,
                           const int *x_tr_lens,
                           const double *data_y_tr,
                           const EdgePair *edges,
                           const double *weights,
                           int data_m,
                           int data_n,
                           int data_p,
                           int para_sparsity,
                           int para_b,
                           double para_xi,
                           double para_l2_reg,
                           int para_num_passes,
                           int para_step_len,
                           int para_verbose,
                           double *re_wt,
                           double *re_wt_bar,
                           double *re_auc) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);
    memset(re_wt, 0, sizeof(double) * data_p); // wt --> 0.0
    memset(re_wt_bar, 0, sizeof(double) * data_p); // wt_bar --> 0.0
    double *grad_wt = malloc(sizeof(double) * data_p); // gradient
    double *u = malloc(sizeof(double) * data_p); // proxy vector
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * data_p);
    memset(posi_x_mean, 0, sizeof(double) * data_p);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * data_p);
    memset(nega_x_mean, 0, sizeof(double) * data_p);
    double alpha_wt; // initialize alpha_wt (initialize to zero.)
    double posi_t = 0.0, nega_t = 0.0; // to determine a_wt, b_wt, and alpha_wt
    double *xt = malloc(sizeof(double) * data_p);
    for (int i = 0; i < data_n; i++) {
        const int *xt_indices = x_tr_indices + x_tr_posis[i];
        const double *xt_vals = x_tr_vals + x_tr_posis[i];
        memset(xt, 0, sizeof(double) * data_p);
        for (int kk = 0; kk < x_tr_lens[i]; kk++) { xt[xt_indices[kk]] = xt_vals[kk]; }
        if (data_y_tr[i] > 0) {
            posi_t++;
            cblas_dscal(data_p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(data_p, 1. / posi_t, xt, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(data_p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(data_p, 1. / nega_t, xt, 1, nega_x_mean, 1);
        }
    }
    // Pr(y=1), learning rate, initial start time is zero=1.0
    double prob_p = posi_t / (data_n * 1.0), eta_t, t = 1.0;
    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(data_p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(data_p, nega_x_mean, 1, nega_x_mean, 1)));
    }
    double *aver_grad = malloc(sizeof(double) * data_p);
    memset(aver_grad, 0, sizeof(double) * data_p);
    double *proj_prizes = malloc(sizeof(double) * data_p);   // projected prizes.
    double *proj_costs = malloc(sizeof(double) * data_m);    // projected costs.
    GraphStat *graph_stat = make_graph_stat(data_p, data_m);   // head projection paras
    double *y_pred = malloc(sizeof(double) * data_n);
    int index_auc = 0;
    for (int i = 0; i < para_num_passes; i++) {
        for (int j = 0; j < data_n / para_b; j++) { // data_n/para_b is the total number of blocks.

            a_wt = cblas_ddot(data_p, re_wt, 1, posi_x_mean, 1); // update a(wt)
            b_wt = cblas_ddot(data_p, re_wt, 1, nega_x_mean, 1); // update b(wt)
            alpha_wt = b_wt - a_wt; // update alpha(wt)
            eta_t = para_xi / sqrt(t); // current learning rate
            // receive a block of training samples to calculate the gradient
            memset(grad_wt, 0, sizeof(double) * data_p);
            for (int kk = 0; kk < para_b; kk++) {
                int ind = j * para_b + kk; // current ind:
                const int *xt_indices = x_tr_indices + x_tr_posis[ind];
                const double *xt_vals = x_tr_vals + x_tr_posis[ind];
                memset(xt, 0, sizeof(double) * data_p);
                for (int tt = 0; tt < x_tr_lens[ind]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
                double wt_dot = cblas_ddot(data_p, re_wt, 1, xt, 1);
                double weight = data_y_tr[ind] > 0 ? 2. * (1.0 - prob_p) * (wt_dot - a_wt) -
                                                     2. * (1.0 + alpha_wt) * (1.0 - prob_p) :
                                2.0 * prob_p * (wt_dot - b_wt) + 2.0 * (1.0 + alpha_wt) * prob_p;
                cblas_daxpy(data_p, weight, xt, 1, grad_wt, 1); // calculate the gradient
            }
            cblas_dscal(data_p, (t - 1.) / t, aver_grad, 1); // take average of wt --> wt_bar
            cblas_daxpy(data_p, 1. / t, grad_wt, 1, aver_grad, 1);
            cblas_dscal(data_p, 1. / (para_b * 1.0), grad_wt, 1);
            cblas_dcopy(data_p, re_wt, 1, u, 1); // gradient descent: u= wt - eta * grad(wt)
            cblas_daxpy(data_p, -eta_t, grad_wt, 1, u, 1);
            /**
             * ell_2 regularization option proposed in the following paper:
             *
             * @inproceedings{singer2009efficient,
             * title={Efficient learning using forward-backward splitting},
             * author={Singer, Yoram and Duchi, John C},
             * booktitle={Advances in Neural Information Processing Systems},
             * pages={495--503},
             * year={2009}}
             */
            cblas_dscal(data_p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
            //to do graph projection.
            for (int kk = 0; kk < data_p; kk++) { proj_prizes[kk] = u[kk] * u[kk]; }
            int g = 1, sparsity_low = para_sparsity, sparsity_high = para_sparsity + 2;
            int tail_max_iter = 50, verbose = 0;
            head_tail_binsearch(edges, weights, proj_prizes, data_p, data_m, g, -1, sparsity_low,
                                sparsity_high, tail_max_iter, GWPruning, verbose, graph_stat);
            cblas_dscal(data_p, 0.0, re_wt, 1);
            for (int kk = 0; kk < graph_stat->re_nodes->size; kk++) {
                int cur_node = graph_stat->re_nodes->array[kk];
                re_wt[cur_node] = u[cur_node];
            }
            cblas_dscal(data_p, (t - 1.) / t, re_wt_bar, 1); // take average of wt --> wt_bar
            cblas_daxpy(data_p, 1. / t, re_wt, 1, re_wt_bar, 1);
            if ((fmod(t, para_step_len) == 0.)) { // to calculate AUC score and run time
                memset(xt, 0, sizeof(double) * data_p);
                const int *xt_indices = x_tr_indices + x_tr_posis[j];
                const double *xt_vals = x_tr_vals + x_tr_posis[j];
                for (int tt = 0; tt < x_tr_lens[j]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
                re_auc[index_auc++] = _auc_score(data_y_tr, y_pred, data_n);
            }
            t = t + 1.0; // increase time
        }
    }
    free(y_pred);
    free(proj_prizes);
    free(proj_costs);
    free_graph_stat(graph_stat);
    free(aver_grad);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
}


void _algo_opauc(const double *data_x_tr,
                 const double *data_y_tr,
                 int data_n,
                 int data_p,
                 double para_eta,
                 double para_lambda,
                 int para_step_len,
                 int para_verbose,
                 double *re_wt,
                 double *re_wt_bar,
                 double *re_auc,
                 double *re_rts) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);
    double start_time = clock();
    double num_p = 0.0, num_n = 0.0;
    double *center_p = malloc(sizeof(double) * data_p);
    double *center_n = malloc(sizeof(double) * data_p);
    double *cov_p = malloc(sizeof(double) * data_p * data_p);
    double *cov_n = malloc(sizeof(double) * data_p * data_p);
    double *grad_wt = malloc(sizeof(double) * data_p);
    memset(center_p, 0, sizeof(double) * data_p);
    memset(center_n, 0, sizeof(double) * data_p);
    memset(cov_p, 0, sizeof(double) * data_p * data_p);
    memset(cov_n, 0, sizeof(double) * data_p * data_p);
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);

    double *tmp_mat = malloc(sizeof(double) * data_p * data_p);
    double *tmp_vec = malloc(sizeof(double) * data_p);
    memset(tmp_mat, 0, sizeof(double) * data_p);
    memset(tmp_vec, 0, sizeof(double) * data_p);
    double *y_pred = malloc(sizeof(double) * data_n);
    int auc_index = 0;
    if (para_verbose > 0) { printf("%d %d\n", data_n, data_p); }
    for (int t = 0; t < data_n; t++) {
        const double *cur_x = data_x_tr + t * data_p;
        double cur_y = data_y_tr[t];
        if (cur_y > 0) {
            num_p++;
            cblas_dcopy(data_p, center_p, 1, tmp_vec, 1); // copy previous center
            cblas_dscal(data_p, (num_p - 1.) / num_p, center_p, 1); // update center_p
            cblas_daxpy(data_p, 1. / num_p, cur_x, 1, center_p, 1);
            cblas_dscal(data_p * data_p, (num_p - 1.) / num_p, cov_p,
                        1); // update covariance matrix
            cblas_dger(CblasRowMajor, data_p, data_p, 1. / num_p, cur_x, 1, cur_x, 1, cov_p,
                       data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, (num_p - 1.) / num_p,
                       tmp_vec, 1, tmp_vec, 1, cov_p, data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, -1., center_p, 1, center_p, 1, cov_p,
                       data_p);
            if (num_n > 0.0) {
                // calculate the gradient part 1: \para_lambda w + x_t - c_t^+
                cblas_dcopy(data_p, center_n, 1, grad_wt, 1);
                cblas_daxpy(data_p, -1., cur_x, 1, grad_wt, 1);
                cblas_daxpy(data_p, para_lambda, re_wt, 1, grad_wt, 1);
                cblas_dcopy(data_p, cur_x, 1, tmp_vec, 1); // xt - c_t^-
                cblas_daxpy(data_p, -1., center_n, 1, tmp_vec, 1);
                cblas_dscal(data_p * data_p, 0.0, tmp_mat, 1); // (xt - c_t^+)(xt - c_t^+)^T
                cblas_dger(CblasRowMajor, data_p, data_p, 1., tmp_vec, 1, tmp_vec, 1, tmp_mat,
                           data_p);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., tmp_mat, data_p,
                            re_wt, 1, 1.0, grad_wt,
                            1);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., cov_n, data_p, re_wt,
                            1, 1.0, grad_wt,
                            1);
            } else {
                cblas_dscal(data_p, 0.0, grad_wt, 1);
            }
        } else {
            num_n++;
            cblas_dcopy(data_p, center_n, 1, tmp_vec, 1); // copy previous center
            cblas_dscal(data_p, (num_n - 1.) / num_n, center_n, 1); // update center_n
            cblas_daxpy(data_p, 1. / num_n, cur_x, 1, center_n, 1);
            cblas_dscal(data_p * data_p, (num_n - 1.) / num_n, cov_n,
                        1); // update covariance matrix
            cblas_dger(CblasRowMajor, data_p, data_p, 1. / num_n, cur_x, 1, cur_x, 1, cov_n,
                       data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, (num_n - 1.) / num_n,
                       tmp_vec, 1, tmp_vec, 1, cov_n, data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, -1., center_n, 1, center_n, 1, cov_n,
                       data_p);
            if (num_p > 0.0) {
                // calculate the gradient part 1: \para_lambda w + x_t - c_t^+
                cblas_dcopy(data_p, cur_x, 1, grad_wt, 1);
                cblas_daxpy(data_p, -1., center_p, 1, grad_wt, 1);
                cblas_daxpy(data_p, para_lambda, re_wt, 1, grad_wt, 1);
                cblas_dcopy(data_p, cur_x, 1, tmp_vec, 1); // xt - c_t^+
                cblas_daxpy(data_p, -1., center_p, 1, tmp_vec, 1);
                cblas_dscal(data_p * data_p, 0.0, tmp_mat, 1); // (xt - c_t^+)(xt - c_t^+)^T
                cblas_dger(CblasRowMajor, data_p, data_p, 1., tmp_vec, 1, tmp_vec, 1, tmp_mat,
                           data_p);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., tmp_mat, data_p,
                            re_wt, 1, 1.0, grad_wt,
                            1);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., cov_p, data_p, re_wt,
                            1, 1.0, grad_wt,
                            1);
            } else {
                cblas_dscal(data_p, 0.0, grad_wt, 1);
            }
        }
        cblas_daxpy(data_p, -para_eta, grad_wt, 1, re_wt, 1); // update the solution
        cblas_dscal(data_p, (t * 1.) / (t * 1. + 1.), re_wt_bar, 1);
        cblas_daxpy(data_p, 1. / (t + 1.), re_wt, 1, re_wt_bar, 1);
        if ((fmod(t, para_step_len) == 0.)) { // to calculate AUC score
            double t_eval = clock();
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
            re_auc[auc_index] = _auc_score(data_y_tr, y_pred, data_n);
            t_eval = clock() - t_eval;
            re_rts[auc_index++] = (clock() - start_time - t_eval) / CLOCKS_PER_SEC;
        }
    }
    free(tmp_vec);
    free(tmp_mat);
    free(grad_wt);
    free(cov_n);
    free(cov_p);
    free(center_n);
    free(center_p);
}


void _algo_opauc_sparse(const double *x_tr_vals,
                        const int *x_tr_inds,
                        const int *x_tr_posis,
                        const int *x_tr_lens,
                        const double *data_y_tr,
                        int data_n,
                        int data_p,
                        int para_tau,
                        double para_eta,
                        double para_lambda,
                        double para_step_len,
                        double para_verbose,
                        double *re_wt,
                        double *re_wt_bar,
                        double *re_auc,
                        double *re_rts) {

    openblas_set_num_threads(1);
    double start_time = clock();
    double num_p = 0.0, num_n = 0.0;
    double *center_p = malloc(sizeof(double) * data_p);
    double *center_n = malloc(sizeof(double) * data_p);
    double *h_center_p = malloc(sizeof(double) * para_tau);
    double *h_center_n = malloc(sizeof(double) * para_tau);
    double *z_p = malloc(sizeof(double) * data_p * para_tau);
    double *z_n = malloc(sizeof(double) * data_p * para_tau);
    double *grad_wt = malloc(sizeof(double) * data_p);
    memset(center_p, 0, sizeof(double) * data_p);
    memset(center_n, 0, sizeof(double) * data_p);
    memset(z_p, 0, sizeof(double) * data_p * data_p);
    memset(z_n, 0, sizeof(double) * data_p * data_p);
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);

    double *tmp_mat = malloc(sizeof(double) * data_p * para_tau);
    double *tmp_vec = malloc(sizeof(double) * data_p);
    memset(tmp_mat, 0, sizeof(double) * data_p);
    memset(tmp_vec, 0, sizeof(double) * data_p);
    double *y_pred = malloc(sizeof(double) * data_n);
    double *xt = malloc(sizeof(double) * data_p);
    double *gaussian_samples = malloc(sizeof(double) * para_tau);
    int auc_index = 0;
    if (para_verbose > 0) { printf("%d %d\n", data_n, data_p); }
    for (int t = 0; t < data_n; t++) {
        const int *xt_indices = x_tr_inds + x_tr_posis[t];
        const double *xt_vals = x_tr_vals + x_tr_posis[t];
        memset(xt, 0, sizeof(double) * data_p);
        for (int tt = 0; tt < x_tr_lens[t]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
        randn(para_tau, gaussian_samples);
        cblas_dscal(para_tau, 1. / sqrt(para_tau), gaussian_samples);
        if (data_y_tr[t] > 0) {
            num_p++;
            cblas_dcopy(data_p, center_p, 1, tmp_vec, 1); // copy previous center
            cblas_dscal(data_p, (num_p - 1.) / num_p, center_p, 1); // update center_p
            cblas_daxpy(data_p, 1. / num_p, xt, 1, center_p, 1);
            cblas_dscal(data_p * data_p, (num_p - 1.) / num_p, cov_p,
                        1); // update covariance matrix
            cblas_dger(CblasRowMajor, data_p, data_p, 1. / num_p, xt, 1, xt, 1, cov_p,
                       data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, (num_p - 1.) / num_p,
                       tmp_vec, 1, tmp_vec, 1, cov_p, data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, -1., center_p, 1, center_p, 1, cov_p,
                       data_p);
            if (num_n > 0.0) {
                // calculate the gradient part 1: \para_lambda w + x_t - c_t^+
                cblas_dcopy(data_p, center_n, 1, grad_wt, 1);
                cblas_daxpy(data_p, -1., xt, 1, grad_wt, 1);
                cblas_daxpy(data_p, para_lambda, re_wt, 1, grad_wt, 1);
                cblas_dcopy(data_p, xt, 1, tmp_vec, 1); // xt - c_t^-
                cblas_daxpy(data_p, -1., center_n, 1, tmp_vec, 1);
                cblas_dscal(data_p * data_p, 0.0, tmp_mat, 1); // (xt - c_t^+)(xt - c_t^+)^T
                cblas_dger(CblasRowMajor, data_p, data_p, 1., tmp_vec, 1, tmp_vec, 1, tmp_mat,
                           data_p);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., tmp_mat, data_p,
                            re_wt, 1, 1.0, grad_wt,
                            1);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., cov_n, data_p, re_wt,
                            1, 1.0, grad_wt,
                            1);
            } else {
                cblas_dscal(data_p, 0.0, grad_wt, 1);
            }
        } else {
            num_n++;
            cblas_dcopy(data_p, center_n, 1, tmp_vec, 1); // copy previous center
            cblas_dscal(data_p, (num_n - 1.) / num_n, center_n, 1); // update center_n
            cblas_daxpy(data_p, 1. / num_n, xt, 1, center_n, 1);
            cblas_dscal(data_p * data_p, (num_n - 1.) / num_n, cov_n,
                        1); // update covariance matrix
            cblas_dger(CblasRowMajor, data_p, data_p, 1. / num_n, xt, 1, xt, 1, cov_n,
                       data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, (num_n - 1.) / num_n,
                       tmp_vec, 1, tmp_vec, 1, cov_n, data_p);
            cblas_dger(CblasRowMajor, data_p, data_p, -1., center_n, 1, center_n, 1, cov_n,
                       data_p);
            if (num_p > 0.0) {
                // calculate the gradient part 1: \para_lambda w + x_t - c_t^+
                cblas_dcopy(data_p, xt, 1, grad_wt, 1);
                cblas_daxpy(data_p, -1., center_p, 1, grad_wt, 1);
                cblas_daxpy(data_p, para_lambda, re_wt, 1, grad_wt, 1);
                cblas_dcopy(data_p, xt, 1, tmp_vec, 1); // xt - c_t^+
                cblas_daxpy(data_p, -1., center_p, 1, tmp_vec, 1);
                cblas_dscal(data_p * data_p, 0.0, tmp_mat, 1); // (xt - c_t^+)(xt - c_t^+)^T
                cblas_dger(CblasRowMajor, data_p, data_p, 1., tmp_vec, 1, tmp_vec, 1, tmp_mat,
                           data_p);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., tmp_mat, data_p,
                            re_wt, 1, 1.0, grad_wt,
                            1);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, data_p, data_p, 1., cov_p, data_p, re_wt,
                            1, 1.0, grad_wt,
                            1);
            } else {
                cblas_dscal(data_p, 0.0, grad_wt, 1);
            }
        }
        cblas_daxpy(data_p, -para_eta, grad_wt, 1, re_wt, 1); // update the solution
        cblas_dscal(data_p, (t * 1.) / (t * 1. + 1.), re_wt_bar, 1);
        cblas_daxpy(data_p, 1. / (t + 1.), re_wt, 1, re_wt_bar, 1);
        if ((fmod(t, para_step_len) == 0.)) { // to calculate AUC score
            double t_eval = clock();
            for (int q = 0; q < data_n; q++) {
                memset(xt, 0, sizeof(double) * data_p);
                xt_indices = x_tr_inds + x_tr_posis[q];
                xt_vals = x_tr_vals + x_tr_posis[q];
                for (int tt = 0; tt < x_tr_lens[q]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
                y_pred[q] = cblas_ddot(data_p, xt, 1, re_wt, 1);
            }
            re_auc[auc_index] = _auc_score(data_y_tr, y_pred, data_n);
            cblas_dgemv(CblasRowMajor, CblasNoTrans,
                        data_n, data_p, 1., xt, data_p, re_wt, 1, 0.0, y_pred, 1);
            re_auc[auc_index] = _auc_score(data_y_tr, y_pred, data_n);
            t_eval = clock() - t_eval;
            re_rts[auc_index++] = (clock() - start_time - t_eval) / CLOCKS_PER_SEC;
        }
    }
    free(gaussian_samples);
    free(xt);
    free(y_pred);
    free(tmp_vec);
    free(tmp_mat);
    free(grad_wt);
    free(cov_n);
    free(cov_p);
    free(h_center_n);
    free(h_center_p);
    free(center_n);
    free(center_p);

}


void _algo_fsauc(const double *data_x_tr,
                 const double *data_y_tr,
                 int data_n,
                 int data_p,
                 double para_r,
                 double para_g,
                 int para_num_passes,
                 int para_step_len,
                 int para_verbose,
                 double *re_wt,
                 double *re_wt_bar,
                 double *re_auc,
                 double *re_rts) {

    openblas_set_num_threads(1); // make sure openblas uses only one cpu at a time.
    double start_time = clock(), delta = 0.1, eta = para_g, R = para_r;
    if (para_verbose > 0) { printf("eta: %.12f R: %.12f", eta, R); }
    double n_ids = para_num_passes * data_n;
    double *v_1 = malloc(sizeof(double) * (data_p + 2)), alpha_1 = 0.0, alpha;
    memset(v_1, 0, sizeof(double) * (data_p + 2));
    double *sx_pos = malloc(sizeof(double) * (data_p));
    memset(sx_pos, 0, sizeof(double) * data_p);
    double *sx_neg = malloc(sizeof(double) * (data_p));
    memset(sx_neg, 0, sizeof(double) * data_p);
    double *m_pos = malloc(sizeof(double) * (data_p));
    memset(m_pos, 0, sizeof(double) * data_p);
    double *m_neg = malloc(sizeof(double) * (data_p));
    memset(m_neg, 0, sizeof(double) * data_p);
    int m = (int) floor(0.5 * log2(2 * n_ids / log2(n_ids))) - 1;
    int n_0 = (int) floor(n_ids / m);
    para_r = 2. * sqrt(3.) * R;
    double p_hat = 0.0, beta = 9.0, D = 2. * sqrt(2.) * para_r, sp = 0.0, t = 0.0;;
    double *gd = malloc(sizeof(double) * (data_p + 2)), gd_alpha;
    memset(gd, 0, sizeof(double) * (data_p + 2));
    double *v_ave = malloc(sizeof(double) * (data_p + 2));
    memset(v_ave, 0, sizeof(double) * (data_p + 2));
    double *v_sum = malloc(sizeof(double) * (data_p + 2));
    double *v = malloc(sizeof(double) * (data_p + 2));
    double *vd = malloc(sizeof(double) * (data_p + 2)), ad;
    double *tmp_proj = malloc(sizeof(double) * data_p), beta_new;
    double *y_pred = malloc(sizeof(double) * data_n);
    int auc_index = 0;
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    for (int k = 0; k < m; k++) {
        memset(v_sum, 0, sizeof(double) * (data_p + 2));
        cblas_dcopy(data_p + 2, v_1, 1, v, 1);
        alpha = alpha_1;
        for (int kk = 0; kk < n_0; kk++) {
            int ind = (k * n_0 + kk) % data_n;
            const double *xt = data_x_tr + ind * data_p;
            double yt = data_y_tr[ind];
            double wx = cblas_ddot(data_p, xt, 1, v, 1);
            double is_posi_y = is_posi(yt), is_nega_y = is_nega(yt);
            sp = sp + is_posi_y;
            p_hat = sp / (t + 1.);
            cblas_daxpy(data_p, is_posi_y, xt, 1, sx_pos, 1);
            cblas_daxpy(data_p, is_nega_y, xt, 1, sx_neg, 1);
            double weight = (1. - p_hat) * (wx - v[data_p] - 1. - alpha) * is_posi_y;
            weight += p_hat * (wx - v[data_p + 1] + 1. + alpha) * is_nega_y;
            cblas_dcopy(data_p, xt, 1, gd, 1);
            cblas_dscal(data_p, weight, gd, 1);
            gd[data_p] = (p_hat - 1.) * (wx - v[data_p]) * is_posi_y;
            gd[data_p + 1] = p_hat * (v[data_p + 1] - wx) * is_nega_y;
            gd_alpha = (p_hat - 1.) * (wx + p_hat * alpha) * is_posi_y +
                       p_hat * (wx + (p_hat - 1.) * alpha) * is_nega_y;
            cblas_daxpy(data_p + 2, -eta, gd, 1, v, 1);
            alpha = alpha + eta * gd_alpha;
            _l1ballproj_condat(v, tmp_proj, data_p, R); //projection to l1-ball
            cblas_dcopy(data_p, tmp_proj, 1, v, 1);
            if (fabs(v[data_p]) > R) { v[data_p] = v[data_p] * (R / fabs(v[data_p])); }
            if (fabs(v[data_p + 1]) > R) {
                v[data_p + 1] = v[data_p + 1] * (R / fabs(v[data_p + 1]));
            }
            if (fabs(alpha) > 2. * R) { alpha = alpha * (2. * R / fabs(alpha)); }
            cblas_dcopy(data_p + 2, v, 1, vd, 1);
            cblas_daxpy(data_p + 2, -1., v_1, 1, vd, 1);
            double norm_vd = sqrt(cblas_ddot(data_p + 2, vd, 1, vd, 1));
            if (norm_vd > para_r) { cblas_dscal(data_p + 2, para_r / norm_vd, vd, 1); }
            cblas_dcopy(data_p + 2, vd, 1, v, 1);
            cblas_daxpy(data_p + 2, 1., v_1, 1, v, 1);
            ad = alpha - alpha_1;
            if (fabs(ad) > D) { ad = ad * (D / fabs(ad)); }
            alpha = alpha_1 + ad;
            cblas_daxpy(data_p + 2, 1., v, 1, v_sum, 1);
            cblas_dcopy(data_p + 2, v_sum, 1, v_ave, 1);
            cblas_dscal(data_p + 2, 1. / (kk + 1.), v_ave, 1);
            t = t + 1.0;
            if ((fmod(t, para_step_len) == 0.)) { // to calculate AUC score
                double t_eval = clock();
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            data_n, data_p, 1., data_x_tr, data_p, re_wt, 1, 0.0, y_pred, 1);
                double auc = _auc_score(data_y_tr, y_pred, data_n);
                t_eval = clock() - t_eval;
                re_auc[auc_index] = auc;
                re_rts[auc_index++] = (clock() - start_time - t_eval) / CLOCKS_PER_SEC;
            }
        }
        para_r = para_r / 2.;
        double tmp1 = 12. * sqrt(2.) * (2. + sqrt(2. * log(12. / delta))) * R;
        double tmp2 = fmin(p_hat, 1. - p_hat) * n_0 - sqrt(2. * n_0 * log(12. / delta));
        if (tmp2 > 0) { D = 2. * sqrt(2.) * para_r + tmp1 / sqrt(tmp2); } else { D = 1e7; }
        tmp1 = 288. * (pow(2. + sqrt(2. * log(12 / delta)), 2.));
        tmp2 = fmin(p_hat, 1. - p_hat) - sqrt(2. * log(12 / delta) / n_0);
        if (tmp2 > 0) { beta_new = 9. + tmp1 / tmp2; } else { beta_new = 1e7; }
        eta = fmin(sqrt(beta_new / beta) * eta / 2, eta);
        beta = beta_new;
        if (sp > 0.0) {
            cblas_dcopy(data_p, sx_pos, 1, m_pos, 1);
            cblas_dscal(data_p, 1. / sp, m_pos, 1);
        }
        if (sp < t) {
            cblas_dcopy(data_p, sx_neg, 1, m_neg, 1);
            cblas_dscal(data_p, 1. / (t - sp), m_neg, 1);
        }
        cblas_dcopy(data_p + 2, v_ave, 1, v_1, 1);
        cblas_dcopy(data_p, m_neg, 1, tmp_proj, 1);
        cblas_daxpy(data_p, -1., m_pos, 1, tmp_proj, 1);
        alpha_1 = cblas_ddot(data_p, v_ave, 1, tmp_proj, 1);
        // wt_bar update
        cblas_dscal(data_p, k / (k + 1.), re_wt_bar, 1);
        cblas_daxpy(data_p, 1. / (k + 1.), v_ave, 1, re_wt_bar, 1);
    }
    cblas_dcopy(data_p, v_ave, 1, re_wt, 1);
    free(y_pred);
    free(tmp_proj);
    free(vd);
    free(v);
    free(v_sum);
    free(v_ave);
    free(gd);
    free(m_neg);
    free(m_pos);
    free(sx_neg);
    free(sx_pos);
    free(v_1);
}

void _algo_fsauc_sparse(const double *x_tr_vals,
                        const int *x_tr_indices,
                        const int *x_tr_posis,
                        const int *x_tr_lens,
                        const double *data_y_tr,
                        int data_n,
                        int data_p,
                        double para_r,
                        double para_g,
                        int para_num_passes,
                        int para_step_len,
                        int para_verbose,
                        double *re_wt,
                        double *re_wt_bar,
                        double *re_auc,
                        double *re_rts) {

    openblas_set_num_threads(1); // make sure openblas uses only one cpu at a time.
    double start_time = clock(), delta = 0.1, eta = para_g, R = para_r;
    if (para_verbose > 0) { printf("eta: %.12f R: %.12f", eta, R); }
    double n_ids = para_num_passes * data_n;
    double *v_1 = malloc(sizeof(double) * (data_p + 2)), alpha_1 = 0.0, alpha;
    memset(v_1, 0, sizeof(double) * (data_p + 2));
    double *sx_pos = malloc(sizeof(double) * (data_p));
    memset(sx_pos, 0, sizeof(double) * data_p);
    double *sx_neg = malloc(sizeof(double) * (data_p));
    memset(sx_neg, 0, sizeof(double) * data_p);
    double *m_pos = malloc(sizeof(double) * (data_p));
    memset(m_pos, 0, sizeof(double) * data_p);
    double *m_neg = malloc(sizeof(double) * (data_p));
    memset(m_neg, 0, sizeof(double) * data_p);
    int m = (int) floor(0.5 * log2(2 * n_ids / log2(n_ids))) - 1;
    int n_0 = (int) floor(n_ids / m);
    para_r = 2. * sqrt(3.) * R;
    double p_hat = 0.0, beta = 9.0, D = 2. * sqrt(2.) * para_r, sp = 0.0, t = 0.0;;
    double *gd = malloc(sizeof(double) * (data_p + 2)), gd_alpha;
    memset(gd, 0, sizeof(double) * (data_p + 2));
    double *v_ave = malloc(sizeof(double) * (data_p + 2));
    memset(v_ave, 0, sizeof(double) * (data_p + 2));
    double *v_sum = malloc(sizeof(double) * (data_p + 2));
    double *v = malloc(sizeof(double) * (data_p + 2));
    double *vd = malloc(sizeof(double) * (data_p + 2)), ad;
    double *tmp_proj = malloc(sizeof(double) * data_p), beta_new;
    double *y_pred = malloc(sizeof(double) * data_n);
    double *xt = malloc(sizeof(double) * data_p);
    int auc_index = 0;
    memset(re_wt, 0, sizeof(double) * data_p);
    memset(re_wt_bar, 0, sizeof(double) * data_p);
    for (int k = 0; k < m; k++) {
        memset(v_sum, 0, sizeof(double) * (data_p + 2));
        cblas_dcopy(data_p + 2, v_1, 1, v, 1);
        alpha = alpha_1;
        for (int kk = 0; kk < n_0; kk++) {
            int ind = (k * n_0 + kk) % data_n;
            const int *xt_indices = x_tr_indices + x_tr_posis[ind];
            const double *xt_vals = x_tr_vals + x_tr_posis[ind];
            memset(xt, 0, sizeof(double) * data_p);
            for (int tt = 0; tt < x_tr_lens[ind]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
            double yt = data_y_tr[ind];
            double wx = cblas_ddot(data_p, xt, 1, v, 1);
            double is_posi_y = is_posi(yt), is_nega_y = is_nega(yt);
            sp = sp + is_posi_y;
            p_hat = sp / (t + 1.);
            cblas_daxpy(data_p, is_posi_y, xt, 1, sx_pos, 1);
            cblas_daxpy(data_p, is_nega_y, xt, 1, sx_neg, 1);
            double weight = (1. - p_hat) * (wx - v[data_p] - 1. - alpha) * is_posi_y;
            weight += p_hat * (wx - v[data_p + 1] + 1. + alpha) * is_nega_y;
            cblas_dcopy(data_p, xt, 1, gd, 1);
            cblas_dscal(data_p, weight, gd, 1);
            gd[data_p] = (p_hat - 1.) * (wx - v[data_p]) * is_posi_y;
            gd[data_p + 1] = p_hat * (v[data_p + 1] - wx) * is_nega_y;
            gd_alpha = (p_hat - 1.) * (wx + p_hat * alpha) * is_posi_y +
                       p_hat * (wx + (p_hat - 1.) * alpha) * is_nega_y;
            cblas_daxpy(data_p + 2, -eta, gd, 1, v, 1);
            alpha = alpha + eta * gd_alpha;
            _l1ballproj_condat(v, tmp_proj, data_p, R); //projection to l1-ball
            cblas_dcopy(data_p, tmp_proj, 1, v, 1);
            if (fabs(v[data_p]) > R) { v[data_p] = v[data_p] * (R / fabs(v[data_p])); }
            if (fabs(v[data_p + 1]) > R) {
                v[data_p + 1] = v[data_p + 1] * (R / fabs(v[data_p + 1]));
            }
            if (fabs(alpha) > 2. * R) { alpha = alpha * (2. * R / fabs(alpha)); }
            cblas_dcopy(data_p + 2, v, 1, vd, 1);
            cblas_daxpy(data_p + 2, -1., v_1, 1, vd, 1);
            double norm_vd = sqrt(cblas_ddot(data_p + 2, vd, 1, vd, 1));
            if (norm_vd > para_r) { cblas_dscal(data_p + 2, para_r / norm_vd, vd, 1); }
            cblas_dcopy(data_p + 2, vd, 1, v, 1);
            cblas_daxpy(data_p + 2, 1., v_1, 1, v, 1);
            ad = alpha - alpha_1;
            if (fabs(ad) > D) { ad = ad * (D / fabs(ad)); }
            alpha = alpha_1 + ad;
            cblas_daxpy(data_p + 2, 1., v, 1, v_sum, 1);
            cblas_dcopy(data_p + 2, v_sum, 1, v_ave, 1);
            cblas_dscal(data_p + 2, 1. / (kk + 1.), v_ave, 1);
            t = t + 1.0;
            if ((fmod(t, para_step_len) == 0.)) { // to calculate AUC score
                double t_eval = clock();
                for (int q = 0; q < data_n; q++) {
                    memset(xt, 0, sizeof(double) * data_p);
                    xt_indices = x_tr_indices + x_tr_posis[q];
                    xt_vals = x_tr_vals + x_tr_posis[q];
                    for (int tt = 0; tt < x_tr_lens[q]; tt++) { xt[xt_indices[tt]] = xt_vals[tt]; }
                    y_pred[q] = cblas_ddot(data_p, xt, 1, re_wt, 1);
                }
                double auc = _auc_score(data_y_tr, y_pred, data_n);
                t_eval = clock() - t_eval;
                re_auc[auc_index] = auc;
                re_rts[auc_index++] = (clock() - start_time - t_eval) / CLOCKS_PER_SEC;
            }
        }
        para_r = para_r / 2.;
        double tmp1 = 12. * sqrt(2.) * (2. + sqrt(2. * log(12. / delta))) * R;
        double tmp2 = fmin(p_hat, 1. - p_hat) * n_0 - sqrt(2. * n_0 * log(12. / delta));
        if (tmp2 > 0) { D = 2. * sqrt(2.) * para_r + tmp1 / sqrt(tmp2); } else { D = 1e7; }
        tmp1 = 288. * (pow(2. + sqrt(2. * log(12 / delta)), 2.));
        tmp2 = fmin(p_hat, 1. - p_hat) - sqrt(2. * log(12 / delta) / n_0);
        if (tmp2 > 0) { beta_new = 9. + tmp1 / tmp2; } else { beta_new = 1e7; }
        eta = fmin(sqrt(beta_new / beta) * eta / 2, eta);
        beta = beta_new;
        if (sp > 0.0) {
            cblas_dcopy(data_p, sx_pos, 1, m_pos, 1);
            cblas_dscal(data_p, 1. / sp, m_pos, 1);
        }
        if (sp < t) {
            cblas_dcopy(data_p, sx_neg, 1, m_neg, 1);
            cblas_dscal(data_p, 1. / (t - sp), m_neg, 1);
        }
        cblas_dcopy(data_p + 2, v_ave, 1, v_1, 1);
        cblas_dcopy(data_p, m_neg, 1, tmp_proj, 1);
        cblas_daxpy(data_p, -1., m_pos, 1, tmp_proj, 1);
        alpha_1 = cblas_ddot(data_p, v_ave, 1, tmp_proj, 1);
        // wt_bar update
        cblas_dscal(data_p, k / (k + 1.), re_wt_bar, 1);
        cblas_daxpy(data_p, 1. / (k + 1.), v_ave, 1, re_wt_bar, 1);
    }
    cblas_dcopy(data_p, v_ave, 1, re_wt, 1);
    free(xt);
    free(y_pred);
    free(tmp_proj);
    free(vd);
    free(v);
    free(v_sum);
    free(v_ave);
    free(gd);
    free(m_neg);
    free(m_pos);
    free(sx_neg);
    free(sx_pos);
    free(v_1);
}
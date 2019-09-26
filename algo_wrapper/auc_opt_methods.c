//
// Created by baojian on 9/9/19.
//
#include "auc_opt_methods.h"


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
 * A full vector y dot product with a sparse vector x.
 *
 * ---
 * x is presented as three elements:
 * 1. x_indices: the nonzero indices.
 * 2. x_values: the nonzeros.
 * 3. x_len: the number of nonzeros.
 * For example,
 * x = {0, 1.0, 0, 0, 0, 0.1, 0.5}, then
 * x_indices = {1,5,6}
 * x_values = {1.0,0.1,0.5}
 * x_len = 3.
 * ---
 *
 * @param x_indices: the nonzero indices of the sparse vector x. starts from 0 index.
 * @param x_values: the nonzeros values of the sparse vector x.
 * @param x_len: the number of nonzeros in sparse vector x.
 * @param y
 * @return
 */
double _sparse_dot(const double *x_values, const int *x_indices, int x_len, const double *y) {
    double result = 0.0;
    for (int i = 0; i < x_len; i++) {
        result += x_values[i] * y[x_indices[i]];
    }
    return result;
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
 * A full vector y + a scaled sparse vector x, i.e., alpha*x + y --> y
 *
 * ---
 * x is presented as three elements:
 * 1. x_indices: the nonzero indices.
 * 2. x_values: the nonzeros.
 * 3. x_len: the number of nonzeros.
 * For example,
 * x = {0, 1.0, 0, 0, 0, 0.1, 0.5}, then
 * x_indices = {1,5,6}
 * x_values = {1.0,0.1,0.5}
 * x_len = 3.
 * ---
 *
 * @param x_indices: the nonzero indices of the sparse vector x. starts from 0 index.
 * @param x_values: the nonzeros values of the sparse vector x.
 * @param x_len: the number of nonzeros in sparse vector x.
 * @param y
 * @return
 */
void _sparse_cblas_daxpy(const double *x_values, const int *x_indices, int x_len,
                         double alpha, double *y) {
    for (int i = 0; i < x_len; i++) {
        y[x_indices[i]] += alpha * x_values[i];
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


bool __solam(const double *x_tr,
             const double *y_tr,
             int num_tr,
             int p,
             double para_r,
             double para_xi,
             int para_num_pass,
             int verbose,
             solam_results *results) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);
    double *zero_v = malloc(sizeof(double) * (p + 2));
    double *one_v = malloc(sizeof(double) * (p + 2));
    for (int i = 0; i < p + 2; i++) {
        zero_v[i] = 0.0;
        one_v[i] = 1.0;
    }
    if (verbose > 0) {
        printf("num_tr: %d p: %d", num_tr, p);
    }
    // start of the algorithm
    double sr = para_r;
    double sc = para_xi;
    double p_hat = 0.; // number of positive
    double *n_v0_ = malloc(sizeof(double) * (p + 2));
    cblas_dcopy(p + 2, zero_v, 1, n_v0_, 1);
    double n_a_p0_ = 0.;
    double n_g_a0_ = 0.;
    // initial vector
    double *n_v0 = malloc(sizeof(double) * (p + 2));
    cblas_dcopy(p, one_v, 1, n_v0, 1);
    cblas_dscal(p, sqrt(sr * sr / (double) p), n_v0, 1);
    n_v0[p] = sr;
    n_v0[p + 1] = sr;
    double n_a_p0 = 2. * sr;
    // iteration time.
    double n_t = 1.;
    int n_cnt = 1;
    double *v_wt = malloc(sizeof(double) * p);
    double *v_p_dv = malloc(sizeof(double) * (p + 2));
    double *n_v1 = malloc(sizeof(double) * (p + 2));
    double *n_v1_ = malloc(sizeof(double) * (p + 2));
    double v_p_da;
    double n_a_p1;
    double n_a_p1_;

    cblas_dcopy(p, n_v0_, 1, results->wt, 1);
    cblas_dcopy(p, n_v0_, 1, results->wt_bar, 1);

    while (true) {
        if (n_cnt > para_num_pass) {
            break;
        }
        for (int j = 0; j < num_tr; j++) {
            // current learning rate
            double n_ga = sc / sqrt(n_t);
            // current sample
            const double *xt = x_tr + j * p;
            double yt = y_tr[j];
            double is_p_yt = is_posi(yt);
            double is_n_yt = is_nega(yt);
            // update p_hat
            p_hat = ((n_t - 1.) * p_hat + is_p_yt) / n_t;
            // update w, a, b, alpha
            cblas_dcopy(p, n_v0, 1, v_wt, 1);
            double n_a = n_v0[p];
            double n_b = n_v0[p + 1];
            // calculate the gradient w
            cblas_dcopy(p + 2, zero_v, 1, v_p_dv, 1);
            double vt_dot = cblas_ddot(p, v_wt, 1, xt, 1);
            double wei_posi = 2. * (1. - p_hat) * (vt_dot - n_a - (1. + n_a_p0));
            double wei_nega = 2. * p_hat * ((vt_dot - n_b) + (1. + n_a_p0));
            double weight = wei_posi * is_p_yt + wei_nega * is_n_yt;
            cblas_daxpy(p, weight, xt, 1, v_p_dv, 1);
            //gradient of a
            v_p_dv[p] = -2. * (1. - p_hat) * (vt_dot - n_a) * is_p_yt;
            //gradient of b
            v_p_dv[p + 1] = -2. * p_hat * (vt_dot - n_b) * is_n_yt;
            // gradient descent step of w
            cblas_dscal(p + 2, -n_ga, v_p_dv, 1);
            cblas_daxpy(p + 2, 1.0, n_v0, 1, v_p_dv, 1);
            // calculate the gradient of dual alpha
            wei_posi = -2. * (1. - p_hat) * vt_dot;
            wei_nega = 2. * p_hat * vt_dot;
            v_p_da = wei_posi * is_p_yt + wei_nega * is_n_yt - 2. * p_hat * (1. - p_hat) * n_a_p0;
            // gradient descent step of alpha
            v_p_da = n_a_p0 + n_ga * v_p_da;

            // normalization -- the projection step.
            double n_rv = sqrt(cblas_ddot(p, v_p_dv, 1, v_p_dv, 1));
            if (n_rv > sr) {
                cblas_dscal(p, 1. / n_rv * sr, v_p_dv, 1);
            }
            if (v_p_dv[p] > sr) {
                v_p_dv[p] = sr;
            }
            if (v_p_dv[p + 1] > sr) {
                v_p_dv[p + 1] = sr;
            }
            cblas_dcopy(p + 2, v_p_dv, 1, n_v1, 1); //n_v1 = v_p_dv
            double n_ra = fabs(v_p_da);
            if (n_ra > 2. * sr) {
                n_a_p1 = v_p_da / n_ra * (2. * sr);
            } else {
                n_a_p1 = v_p_da;
            }
            // update gamma_
            double n_g_a1_ = n_g_a0_ + n_ga;
            // update v_
            cblas_dcopy(p + 2, n_v0, 1, n_v1_, 1);
            cblas_dscal(p + 2, n_ga / n_g_a1_, n_v1_, 1);
            cblas_daxpy(p + 2, n_g_a0_ / n_g_a1_, n_v0_, 1, n_v1_, 1);
            // update alpha_
            n_a_p1_ = (n_g_a0_ * n_a_p0_ + n_ga * n_a_p0) / n_g_a1_;
            // update the information
            cblas_dcopy(p + 2, n_v1_, 1, n_v0_, 1); // n_v0_ = n_v1_;
            n_a_p0_ = n_a_p1_;
            n_g_a0_ = n_g_a1_;
            cblas_dcopy(p + 2, n_v1, 1, n_v0, 1); // n_v0 = n_v1;
            n_a_p0 = n_a_p1;
            // updates the results.

            cblas_dscal(p, (n_t - 1.) / n_t, results->wt_bar, 1);
            cblas_daxpy(p, 1. / n_t, n_v1_, 1, results->wt_bar, 1);
            // update the counts
            n_t = n_t + 1.;
        }
        n_cnt += 1;
    }
    cblas_dcopy(p, n_v1_, 1, results->wt, 1);
    results->a = n_v1_[p];
    results->b = n_v1_[p + 1];
    free(n_v1_);
    free(n_v1);
    free(n_v0);
    free(n_v0_);
    free(v_p_dv);
    free(v_wt);
    free(one_v);
    free(zero_v);
    return true;
}

bool __solam_sparse(const double *x_tr_vals,
                    const int *x_tr_indices,
                    const int *x_tr_lens,
                    const int *x_tr_posis,
                    const double *y_tr,
                    int num_tr,
                    int p,
                    double para_r,
                    double para_xi,
                    int para_num_pass,
                    int verbose,
                    solam_results *results) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);

    double *zero_v = malloc(sizeof(double) * (p + 2));
    double *one_v = malloc(sizeof(double) * (p + 2));
    for (int i = 0; i < p + 2; i++) {
        zero_v[i] = 0.0;
        one_v[i] = 1.0;
    }
    if (verbose > 0) {
        printf("num_tr: %d p: %d", num_tr, p);
    }
    // start of the algorithm
    double sr = para_r;
    double sc = para_xi;
    double p_hat = 0.; // number of positive
    double *n_v0_ = malloc(sizeof(double) * (p + 2));
    cblas_dcopy(p + 2, zero_v, 1, n_v0_, 1);
    double n_a_p0_ = 0.;
    double n_g_a0_ = 0.;
    // initial vector
    double *n_v0 = malloc(sizeof(double) * (p + 2));
    cblas_dcopy(p, one_v, 1, n_v0, 1);
    cblas_dscal(p, sqrt(sr * sr / (double) p), n_v0, 1);
    n_v0[p] = sr;
    n_v0[p + 1] = sr;
    double n_a_p0 = 2. * sr;
    // iteration time.
    double n_t = 1.;
    int n_cnt = 1;
    double *v_wt = malloc(sizeof(double) * p);
    double *v_p_dv = malloc(sizeof(double) * (p + 2));
    double *n_v1 = malloc(sizeof(double) * (p + 2));
    double *n_v1_ = malloc(sizeof(double) * (p + 2));
    double v_p_da;
    double n_a_p1;
    double n_a_p1_;

    cblas_dcopy(p, n_v0_, 1, results->wt, 1);
    cblas_dcopy(p, n_v0_, 1, results->wt_bar, 1);

    double *xt = malloc(sizeof(double) * p);

    while (true) {
        if (n_cnt > para_num_pass) {
            break;
        }
        for (int j = 0; j < num_tr; j++) {
            //get sparse vector to full vector
            const int *xt_indices = x_tr_indices + x_tr_posis[j];
            const double *xt_vals = x_tr_vals + x_tr_posis[j];
            cblas_dcopy(p, zero_v, 1, xt, 1);
            for (int kk = 0; kk < x_tr_lens[j]; kk++) {
                xt[xt_indices[kk]] = xt_vals[kk];
            }
            // current learning rate
            double n_ga = sc / sqrt(n_t);
            // current sample
            double yt = y_tr[j];
            double is_p_yt = is_posi(yt);
            double is_n_yt = is_nega(yt);
            // update p_hat
            p_hat = ((n_t - 1.) * p_hat + is_p_yt) / n_t;
            // update w, a, b, alpha
            cblas_dcopy(p, n_v0, 1, v_wt, 1);
            double n_a = n_v0[p];
            double n_b = n_v0[p + 1];
            // calculate the gradient w
            cblas_dcopy(p + 2, zero_v, 1, v_p_dv, 1);
            double vt_dot = cblas_ddot(p, v_wt, 1, xt, 1);
            double wei_posi = 2. * (1. - p_hat) * (vt_dot - n_a - (1. + n_a_p0));
            double wei_nega = 2. * p_hat * ((vt_dot - n_b) + (1. + n_a_p0));
            double weight = wei_posi * is_p_yt + wei_nega * is_n_yt;
            cblas_daxpy(p, weight, xt, 1, v_p_dv, 1);
            //gradient of a
            v_p_dv[p] = -2. * (1. - p_hat) * (vt_dot - n_a) * is_p_yt;
            //gradient of b
            v_p_dv[p + 1] = -2. * p_hat * (vt_dot - n_b) * is_n_yt;
            // gradient descent step of w
            cblas_dscal(p + 2, -n_ga, v_p_dv, 1);
            cblas_daxpy(p + 2, 1.0, n_v0, 1, v_p_dv, 1);
            // calculate the gradient of dual alpha
            wei_posi = -2. * (1. - p_hat) * vt_dot;
            wei_nega = 2. * p_hat * vt_dot;
            v_p_da = wei_posi * is_p_yt + wei_nega * is_n_yt - 2. * p_hat * (1. - p_hat) * n_a_p0;
            // gradient descent step of alpha
            v_p_da = n_a_p0 + n_ga * v_p_da;

            // normalization -- the projection step.
            double n_rv = sqrt(cblas_ddot(p, v_p_dv, 1, v_p_dv, 1));
            if (n_rv > sr) {
                cblas_dscal(p, 1. / n_rv * sr, v_p_dv, 1);
            }
            if (v_p_dv[p] > sr) {
                v_p_dv[p] = sr;
            }
            if (v_p_dv[p + 1] > sr) {
                v_p_dv[p + 1] = sr;
            }
            cblas_dcopy(p + 2, v_p_dv, 1, n_v1, 1); //n_v1 = v_p_dv
            double n_ra = fabs(v_p_da);
            if (n_ra > 2. * sr) {
                n_a_p1 = v_p_da / n_ra * (2. * sr);
            } else {
                n_a_p1 = v_p_da;
            }
            // update gamma_
            double n_g_a1_ = n_g_a0_ + n_ga;
            // update v_
            cblas_dcopy(p + 2, n_v0, 1, n_v1_, 1);
            cblas_dscal(p + 2, n_ga / n_g_a1_, n_v1_, 1);
            cblas_daxpy(p + 2, n_g_a0_ / n_g_a1_, n_v0_, 1, n_v1_, 1);
            // update alpha_
            n_a_p1_ = (n_g_a0_ * n_a_p0_ + n_ga * n_a_p0) / n_g_a1_;
            // update the information
            cblas_dcopy(p + 2, n_v1_, 1, n_v0_, 1); // n_v0_ = n_v1_;
            n_a_p0_ = n_a_p1_;
            n_g_a0_ = n_g_a1_;
            cblas_dcopy(p + 2, n_v1, 1, n_v0, 1); // n_v0 = n_v1;
            n_a_p0 = n_a_p1;
            // updates the results.
            cblas_dscal(p, (n_t - 1.) / n_t, results->wt_bar, 1);
            cblas_daxpy(p, 1. / n_t, n_v1_, 1, results->wt_bar, 1);
            // update the counts
            n_t = n_t + 1.;
        }
        n_cnt += 1;
    }
    cblas_dcopy(p, n_v1_, 1, results->wt, 1);
    results->a = n_v1_[p];
    results->b = n_v1_[p + 1];
    free(xt);
    free(n_v1_);
    free(n_v1);
    free(n_v0);
    free(n_v0_);
    free(v_p_dv);
    free(v_wt);
    free(one_v);
    free(zero_v);
    return true;
}




/**
 * Stochastic Proximal AUC Maximization with elastic net penalty
 * SPAM algorithm.
 * @param x_tr: The matrix of data samples.
 * @param y_tr: We assume that each y_tr[i] is either +1.0 or -1.0.
 * @param p: >=1 (at least one feature).
 * @param n: >=2 (at least two samples).
 * @param para_num_passes: >=1 (at least pass dataset once)
 * @param para_xi: >0 (constant factor of learning rate).
 * @param para_l1_reg: >=0. (==0.0 without l1-regularization.)
 * @param para_l2_reg: >=0. (==0.0 without l2-regularization.)
 * @param results: wt/wt_bar.
 * @return
 */
bool __spam(const double *x_tr,
            const double *y_tr,
            int p,
            int n,
            double para_xi,
            double para_l1_reg,
            double para_l2_reg,
            int para_num_passes,
            int para_step_len,
            int para_reg_opt,
            int para_verbose,
            spam_results *results) {
    // start time clock
    double start_time = clock();

    // zero vector and set it  to zero.
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, sizeof(double) * p);

    // wt --> 0.0
    double *wt = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt, 1);
    // wt_bar --> 0.0
    double *wt_bar = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt_bar, 1);
    // gradient
    double *grad_wt = malloc(sizeof(double) * p);
    // proxy vector
    double *u = malloc(sizeof(double) * p);

    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x_mean, 1);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x_mean, 1);
    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;

    // to determine a_wt, b_wt, and alpha_wt
    double posi_t = 0.0, nega_t = 0.0;
    for (int i = 0; i < n; i++) {
        const double *cur_xt = x_tr + i * p;
        double yt = y_tr[i];
        if (yt > 0) {
            posi_t++;
            cblas_dscal(p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(p, 1. / posi_t, cur_xt, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(p, 1. / nega_t, cur_xt, 1, nega_x_mean, 1);
        }
    }

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (n * 1.0);

    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(p, nega_x_mean, 1, nega_x_mean, 1)));
    }

    // learning rate
    double eta_t;

    // initial start time is zero=1.0
    double t = 1.0;

    //intialize the results
    results->t_index = 0;
    results->t_eval_time = 0.0;

    for (int i = 0; i < para_num_passes; i++) {
        // for each training sample j
        // printf("epoch: %d\n", i);
        for (int j = 0; j < n; j++) {
            // receive training sample zt=(xt,yt)
            const double *cur_xt = x_tr + j * p;
            double cur_yt = y_tr[j];

            // current learning rate
            eta_t = para_xi / sqrt(t);

            // update a(wt), para_b(wt), and alpha(wt)
            a_wt = cblas_ddot(p, wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(p, wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;

            double weight;
            if (cur_yt > 0) {
                weight = 2. * (1.0 - prob_p) * (cblas_ddot(p, wt, 1, cur_xt, 1) - a_wt);
                weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
            } else {
                weight = 2.0 * prob_p * (cblas_ddot(p, wt, 1, cur_xt, 1) - b_wt);
                weight += 2.0 * (1.0 + alpha_wt) * prob_p;
            }
            if (para_verbose > 0) {
                printf("cur_iter: %05d lr: %.4f a_wt: %.4f b_wt: %.4f alpha_wt: %.4f "
                       "weight: %.4f\n", j, eta_t, a_wt, b_wt, alpha_wt, weight);
            }

            // calculate the gradient
            cblas_dcopy(p, cur_xt, 1, grad_wt, 1);
            cblas_dscal(p, weight, grad_wt, 1);

            //gradient descent step: u= wt - eta * grad(wt)
            cblas_dcopy(p, wt, 1, u, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, u, 1);

            // elastic net
            if (para_reg_opt == 0) {
                /**
                 * Currently, u is the \hat{wt_{t+1}}, next is to use prox_operator.
                 * The following part of the code is the proximal operator for elastic norm.
                 *
                 * @inproceedings{singer2009efficient,
                 * title={Efficient learning using forward-backward splitting},
                 * author={Singer, Yoram and Duchi, John C},
                 * booktitle={Advances in Neural Information Processing Systems},
                 * pages={495--503},
                 * year={2009}}
                 */
                double tmp_l2 = (eta_t * para_l2_reg + 1.);
                for (int k = 0; k < p; k++) {
                    double sign_uk = (double) (sign(u[k]));
                    wt[k] = (sign_uk / tmp_l2) * fmax(0.0, fabs(u[k]) - eta_t * para_l1_reg);
                }
            } else {
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
                cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
                cblas_dcopy(p, u, 1, wt, 1);
            }

            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, wt_bar, 1);
            cblas_daxpy(p, 1. / t, wt, 1, wt_bar, 1);

            // to calculate AUC score and run time
            if ((fmod(t, para_step_len) == 0.) || t == 1.) {
                double eval_start = clock();
                double *y_pred = malloc(sizeof(double) * n);
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            n, p, 1., x_tr, p, wt_bar, 1, 0.0, y_pred, 1);
                double auc = _auc_score(y_tr, y_pred, n);
                free(y_pred);
                double eval_time = (clock() - eval_start) / CLOCKS_PER_SEC;
                results->t_eval_time += eval_time;
                double run_time = (clock() - start_time) / CLOCKS_PER_SEC - results->t_eval_time;
                results->t_run_time[results->t_index] = run_time;
                results->t_auc[results->t_index] = auc;
                results->t_indices[results->t_index] = i * n + j;
                results->t_index++;
                if (para_verbose > 0) {
                    printf("current auc score: %.4f, t:  %.f\n", auc, t);
                }
            }
            // increase time
            t++;
        }
    }
    cblas_dcopy(p, wt_bar, 1, results->wt_bar, 1);
    cblas_dcopy(p, wt, 1, results->wt, 1);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
    free(wt_bar);
    free(wt);
    free(zero_vector);
    return true;
}

bool __spam_sparse(const double *x_tr_vals,
                   const int *x_tr_indices,
                   const int *x_tr_posis,
                   const int *x_tr_lens,
                   const double *y_tr,
                   int p,
                   int n,
                   double para_xi,
                   double para_l1_reg,
                   double para_l2_reg,
                   int num_passes,
                   int step_len,
                   int reg_opt,
                   int verbose,
                   spam_results *results) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);

    // start time clock
    double start_time = clock();

    // zero vector and set it  to zero.
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, sizeof(double) * p);

    // wt --> 0.0
    cblas_dcopy(p, zero_vector, 1, results->wt, 1);
    // wt_bar --> 0.0
    cblas_dcopy(p, zero_vector, 1, results->wt_bar, 1);
    // gradient
    double *grad_wt = malloc(sizeof(double) * p);
    // proxy vector
    double *u = malloc(sizeof(double) * p);

    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x_mean, 1);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x_mean, 1);
    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;
    double *xt = malloc(sizeof(double) * p);
    // to determine a_wt, b_wt, and alpha_wt
    double posi_t = 0.0, nega_t = 0.0;

    for (int i = 0; i < n; i++) {
        // get current sample
        const int *xt_indices = x_tr_indices + x_tr_posis[i];
        const double *xt_vals = x_tr_vals + x_tr_posis[i];
        cblas_dcopy(p, zero_vector, 1, xt, 1);
        for (int kk = 0; kk < x_tr_lens[i]; kk++) {
            xt[xt_indices[kk]] = xt_vals[kk];
        }
        double yt = y_tr[i];
        if (yt > 0) {
            posi_t++;
            cblas_dscal(p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(p, 1. / posi_t, xt, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(p, 1. / nega_t, xt, 1, nega_x_mean, 1);
        }
    }

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (n * 1.0);

    if (verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(p, nega_x_mean, 1, nega_x_mean, 1)));
    }

    // learning rate
    double eta_t;

    // initial start time is zero=1.0
    double t = 1.0;

    //initialize the results
    results->t_index = 0;
    results->t_eval_time = 0.0;

    double *tmp_vec = malloc(sizeof(double) * p);

    for (int i = 0; i < num_passes; i++) {
        // for each training sample j
        // printf("epoch: %d\n", i);
        double per_s_time = clock();
        for (int j = 0; j < n; j++) {
            // receive training sample zt=(xt,yt)
            const int *xt_indices = x_tr_indices + x_tr_posis[j];
            const double *xt_vals = x_tr_vals + x_tr_posis[j];
            cblas_dcopy(p, zero_vector, 1, xt, 1);
            for (int tt = 0; tt < x_tr_lens[j]; tt++) {
                xt[xt_indices[tt]] = xt_vals[tt];
            }
            double cur_yt = y_tr[j];

            // current learning rate
            eta_t = para_xi / sqrt(t);

            // update a(wt), para_b(wt), and alpha(wt)
            a_wt = cblas_ddot(p, results->wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(p, results->wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;

            double weight;
            double dot_prod = cblas_ddot(p, xt, 1, results->wt, 1);
            if (cur_yt > 0) {
                weight = 2. * (1.0 - prob_p) * (dot_prod - a_wt);
                weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
            } else {
                weight = 2.0 * prob_p * (dot_prod - b_wt);
                weight += 2.0 * (1.0 + alpha_wt) * prob_p;
            }
            if (verbose > 0) {
                printf("cur_iter: %05d lr: %.4f a_wt: %.4f b_wt: %.4f alpha_wt: %.4f "
                       "weight: %.4f\n", j, eta_t, a_wt, b_wt, alpha_wt, weight);
            }

            // calculate the gradient
            cblas_dcopy(p, xt, 1, grad_wt, 1);
            cblas_dscal(p, weight, grad_wt, 1);

            //gradient descent step: u= wt - eta * grad(wt)
            cblas_dcopy(p, results->wt, 1, u, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, u, 1);

            // elastic net
            if (reg_opt == 0) {
                /**
                 * Currently, u is the \hat{wt_{t+1}}, next is to use prox_operator.
                 * The following part of the code is the proximal operator for elastic norm.
                 *
                 * @inproceedings{singer2009efficient,
                 * title={Efficient learning using forward-backward splitting},
                 * author={Singer, Yoram and Duchi, John C},
                 * booktitle={Advances in Neural Information Processing Systems},
                 * pages={495--503},
                 * year={2009}}
                 */
                double tmp_l2 = (eta_t * para_l2_reg + 1.);
                for (int k = 0; k < p; k++) {
                    double sign_uk = (double) (sign(u[k]));
                    results->wt[k] = (sign_uk / tmp_l2) *
                            fmax(0.0, fabs(u[k]) - eta_t * para_l1_reg);
                }
            } else {
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
                cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
                cblas_dcopy(p, u, 1, results->wt, 1);
            }

            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, results->wt_bar, 1);
            cblas_daxpy(p, 1. / t, results->wt, 1, results->wt_bar, 1);

            // to calculate AUC score and run time
            if (fmod(t, step_len) == 0.) {
                double eval_start = clock();
                double *y_pred = malloc(sizeof(double) * n);
                for (int q = 0; q < n; q++) {
                    cblas_dcopy(p, zero_vector, 1, xt, 1);
                    xt_indices = x_tr_indices + x_tr_posis[j];
                    xt_vals = x_tr_vals + x_tr_posis[j];
                    for (int tt = 0; tt < x_tr_lens[j]; tt++) {
                        xt[xt_indices[tt]] = xt_vals[tt];
                    }
                    y_pred[q] = cblas_ddot(p, xt, 1, results->wt_bar, 1);
                }
                double auc = _auc_score(y_tr, y_pred, n);
                free(y_pred);
                double eval_time = (clock() - eval_start) / CLOCKS_PER_SEC;
                results->t_eval_time += eval_time;
                double run_time = (clock() - start_time) / CLOCKS_PER_SEC - results->t_eval_time;
                results->t_run_time[results->t_index] = run_time;
                results->t_auc[results->t_index] = auc;
                results->t_indices[results->t_index] = i * n + j;
                results->t_index++;
                if (verbose == 0) {
                    printf("current auc score: %.4f\n", auc);
                }
            }
            // increase time
            t++;
        }
        if (verbose > 0) {
            printf("run time: %.4f\n", (clock() - per_s_time) / CLOCKS_PER_SEC);
        }
    }
    free(xt);
    free(tmp_vec);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
    free(zero_vector);
    return true;
}


bool __sht_am(const double *x_tr,
              const double *y_tr,
              int p,
              int n,
              int b,
              double para_xi,
              double para_l2_reg,
              int para_sparsity,
              int para_num_passes,
              int para_step_len,
              int para_verbose,
              sht_am_results *results) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);

    // start time clock
    double start_time = clock();

    // zero vector and set it  to zero.
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, sizeof(double) * p);

    // wt --> 0.0
    double *wt = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt, 1);

    // wt_bar --> 0.0
    double *wt_bar = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt_bar, 1);

    // gradient
    double *grad_wt = malloc(sizeof(double) * p);

    // proxy vector
    double *u = malloc(sizeof(double) * p);

    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x_mean, 1);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x_mean, 1);
    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;

    // to determine a_wt, b_wt, and alpha_wt
    double posi_t = 0.0, nega_t = 0.0;
    for (int i = 0; i < n; i++) {
        const double *cur_xt = x_tr + i * p;
        double yt = y_tr[i];
        if (yt > 0) {
            posi_t++;
            cblas_dscal(p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(p, 1. / posi_t, cur_xt, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(p, 1. / nega_t, cur_xt, 1, nega_x_mean, 1);
        }
    }

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (n * 1.0);

    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(p, nega_x_mean, 1, nega_x_mean, 1)));
    }

    // learning rate
    double eta_t;

    // initial start time is zero=1.0
    double t = 1.0;

    //intialize the results
    results->t_index = 0;
    results->t_eval_time = 0.0;

    double *aver_grad = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, aver_grad, 1);

    for (int i = 0; i < para_num_passes; i++) {
        // for each training sample j
        // printf("epoch: %d\n", i);
        for (int j = 0; j < n / b; j++) { // n/b is the total number of blocks.
            // update a(wt), para_b(wt), and alpha(wt)
            a_wt = cblas_ddot(p, wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(p, wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;

            // current learning rate
            eta_t = para_xi / sqrt(t);

            // receive a block of training samples to calculate the gradient
            cblas_dcopy(p, zero_vector, 1, grad_wt, 1);
            for (int kk = 0; kk < b; kk++) {
                const double *cur_xt = x_tr + j * b * p + kk * p;
                double cur_yt = y_tr[j * b + kk];
                double weight;
                if (cur_yt > 0) {
                    weight = 2. * (1.0 - prob_p) * (cblas_ddot(p, wt, 1, cur_xt, 1) - a_wt);
                    weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
                } else {
                    weight = 2.0 * prob_p * (cblas_ddot(p, wt, 1, cur_xt, 1) - b_wt);
                    weight += 2.0 * (1.0 + alpha_wt) * prob_p;
                }
                if (para_verbose > 0) {
                    printf("cur_iter: %05d lr: %.4f a_wt: %.4f b_wt: %.4f alpha_wt: %.4f "
                           "weight: %.4f\n", j, eta_t, a_wt, b_wt, alpha_wt, weight);
                }
                // calculate the gradient
                cblas_daxpy(p, weight, cur_xt, 1, grad_wt, 1);
            }

            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, aver_grad, 1);
            cblas_daxpy(p, 1. / t, grad_wt, 1, aver_grad, 1);

            cblas_dscal(p, 1. / (b * 1.0), grad_wt, 1);



            //gradient descent step: u= wt - eta * grad(wt)
            cblas_dcopy(p, wt, 1, u, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, u, 1);

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
            cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
            _hard_thresholding(u, p, para_sparsity); // k-sparse step.
            // _hard_thresholding_2(aver_grad, p, para_sparsity, u);
            cblas_dcopy(p, u, 1, wt, 1);
            // cblas_dcopy(p, zero_vector, 1, wt, 1);
            // for (int k = 0; k < para_nodes_len; k++) {
            //     wt[para_nodes[k]] = u[para_nodes[k]];
            // }


            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, wt_bar, 1);
            cblas_daxpy(p, 1. / t, wt, 1, wt_bar, 1);

            // to calculate AUC score and run time
            if ((fmod(t, para_step_len) == 0.)) {
                double eval_start = clock();
                double *y_pred = malloc(sizeof(double) * n);
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            n, p, 1., x_tr, p, wt_bar, 1, 0.0, y_pred, 1);
                double auc = _auc_score(y_tr, y_pred, n);
                free(y_pred);
                double eval_time = (clock() - eval_start) / CLOCKS_PER_SEC;
                results->t_eval_time += eval_time;
                double run_time = (clock() - start_time) / CLOCKS_PER_SEC - results->t_eval_time;
                results->t_run_time[results->t_index] = run_time;
                results->t_auc[results->t_index] = auc;
                results->t_indices[results->t_index] = i * n + j;
                results->t_index++;
                if (para_verbose == 0) {
                    printf("current auc score: %.4f\n", auc);
                }
            }
            // increase time
            t++;
        }
    }
    cblas_dcopy(p, wt_bar, 1, results->wt_bar, 1);
    cblas_dcopy(p, wt, 1, results->wt, 1);
    free(aver_grad);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
    free(wt_bar);
    free(wt);
    free(zero_vector);
    return true;
}


bool __sht_am_sparse(const double *x_tr_vals,// the values of these nonzeros.
                     const int *x_tr_indices,  // the inidices of these nonzeros.
                     const int *x_tr_posis,// the start indices of these nonzeros.
                     const int *x_tr_lens, // the list of sizes of nonzeros.
                     const double *y_tr,    // the vector of training samples.
                     int p,                 // the dimension of the features of the dataset
                     int n,                 // the total number of training samples.
                     int b,
                     double para_xi,
                     double para_l2_reg,
                     int para_sparsity,
                     int para_num_passes,
                     int para_step_len,
                     int para_verbose,
                     sht_am_results *results) {

    // make sure openblas uses only one cpu at a time.
    openblas_set_num_threads(1);

    // start time clock
    double start_time = clock();

    // zero vector and set it  to zero.
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, sizeof(double) * p);

    // wt --> 0.0
    double *wt = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt, 1);

    // wt_bar --> 0.0
    double *wt_bar = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt_bar, 1);

    // gradient
    double *grad_wt = malloc(sizeof(double) * p);

    // proxy vector
    double *u = malloc(sizeof(double) * p);

    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x_mean, 1);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x_mean, 1);
    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;
    double *xt = malloc(sizeof(double) * p);
    // to determine a_wt, b_wt, and alpha_wt
    double posi_t = 0.0, nega_t = 0.0;
    for (int i = 0; i < n; i++) {

        const int *xt_indices = x_tr_indices + x_tr_posis[i];
        const double *xt_vals = x_tr_vals + x_tr_posis[i];
        cblas_dcopy(p, zero_vector, 1, xt, 1);
        for (int kk = 0; kk < x_tr_lens[i]; kk++) {
            xt[xt_indices[kk]] = xt_vals[kk];
        }
        double yt = y_tr[i];
        if (yt > 0) {
            posi_t++;
            cblas_dscal(p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(p, 1. / posi_t, xt, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(p, 1. / nega_t, xt, 1, nega_x_mean, 1);
        }
    }

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (n * 1.0);

    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(p, nega_x_mean, 1, nega_x_mean, 1)));
    }

    // learning rate
    double eta_t;

    // initial start time is zero=1.0
    double t = 1.0;

    //intialize the results
    results->t_index = 0;
    results->t_eval_time = 0.0;

    double *aver_grad = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, aver_grad, 1);


    for (int i = 0; i < para_num_passes; i++) {
        // for each training sample j
        // printf("epoch: %d\n", i);
        for (int j = 0; j < n / b; j++) { // n/b is the total number of blocks.
            // update a(wt), para_b(wt), and alpha(wt)
            a_wt = cblas_ddot(p, wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(p, wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;

            // current learning rate
            eta_t = para_xi / sqrt(t);

            // receive a block of training samples to calculate the gradient
            cblas_dcopy(p, zero_vector, 1, grad_wt, 1);
            for (int kk = 0; kk < b; kk++) {
                // current index:
                int index = j * b + kk;
                const int *xt_indices = x_tr_indices + x_tr_posis[index];
                const double *xt_vals = x_tr_vals + x_tr_posis[index];
                cblas_dcopy(p, zero_vector, 1, xt, 1);
                for (int tt = 0; tt < x_tr_lens[index]; tt++) {
                    xt[xt_indices[tt]] = xt_vals[tt];
                }
                double cur_yt = y_tr[j * b + kk];
                double weight;
                if (cur_yt > 0) {
                    weight = 2. * (1.0 - prob_p) * (cblas_ddot(p, wt, 1, xt, 1) - a_wt);
                    weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
                } else {
                    weight = 2.0 * prob_p * (cblas_ddot(p, wt, 1, xt, 1) - b_wt);
                    weight += 2.0 * (1.0 + alpha_wt) * prob_p;
                }
                if (para_verbose > 0) {
                    printf("cur_iter: %05d lr: %.4f a_wt: %.4f b_wt: %.4f alpha_wt: %.4f "
                           "weight: %.4f\n", j, eta_t, a_wt, b_wt, alpha_wt, weight);
                }
                // calculate the gradient
                cblas_daxpy(p, weight, xt, 1, grad_wt, 1);
            }

            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, aver_grad, 1);
            cblas_daxpy(p, 1. / t, grad_wt, 1, aver_grad, 1);

            cblas_dscal(p, 1. / (b * 1.0), grad_wt, 1);



            //gradient descent step: u= wt - eta * grad(wt)
            cblas_dcopy(p, wt, 1, u, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, u, 1);

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
            cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
            _hard_thresholding(u, p, para_sparsity); // k-sparse step.
            // _hard_thresholding_2(aver_grad, p, para_sparsity, u);
            cblas_dcopy(p, u, 1, wt, 1);
            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, wt_bar, 1);
            cblas_daxpy(p, 1. / t, wt, 1, wt_bar, 1);

            // to calculate AUC score and run time
            if ((fmod(t, para_step_len) == 0.)) {
                //TODO add code here.
                double auc = 0.0;
                if (para_verbose == 0) {
                    printf("current auc score: %.4f\n", auc);
                }
            }
            // increase time
            t++;
        }
    }
    cblas_dcopy(p, wt_bar, 1, results->wt_bar, 1);
    cblas_dcopy(p, wt, 1, results->wt, 1);
    results->t_eval_time = (clock() - start_time) / CLOCKS_PER_SEC;
    free(xt);
    free(aver_grad);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
    free(wt_bar);
    free(wt);
    free(zero_vector);
    return true;
}


bool __graph_am(const double *x_tr,
                const double *y_tr,
                int p,
                int n,
                int b,
                double para_xi,
                double para_l2_reg,
                int para_sparsity,
                int para_num_passes,
                int para_step_len,
                int para_verbose,
                const EdgePair *edges,
                const double *weights,
                int m,
                graph_am_results *results) {

    // start time clock
    double start_time = clock();

    // zero vector and set it  to zero.
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, sizeof(double) * p);

    // wt --> 0.0
    double *wt = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt, 1);

    // wt_bar --> 0.0
    double *wt_bar = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt_bar, 1);

    // gradient
    double *grad_wt = malloc(sizeof(double) * p);

    // proxy vector
    double *u = malloc(sizeof(double) * p);

    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x_mean, 1);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x_mean, 1);
    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;

    // to determine a_wt, b_wt, and alpha_wt
    double posi_t = 0.0, nega_t = 0.0;
    for (int i = 0; i < n; i++) {
        const double *cur_xt = x_tr + i * p;
        double yt = y_tr[i];
        if (yt > 0) {
            posi_t++;
            cblas_dscal(p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(p, 1. / posi_t, cur_xt, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(p, 1. / nega_t, cur_xt, 1, nega_x_mean, 1);
        }
    }

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (n * 1.0);

    if (para_verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(p, nega_x_mean, 1, nega_x_mean, 1)));
    }

    // learning rate
    double eta_t;

    // initial start time is zero=1.0
    double t = 1.0;

    //intialize the results
    results->t_index = 0;
    results->t_eval_time = 0.0;

    double *aver_grad = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, aver_grad, 1);

    double *proj_prizes = malloc(sizeof(double) * p);   // projected prizes.
    double *proj_costs = malloc(sizeof(double) * m);    // projected costs.
    GraphStat *graph_stat = make_graph_stat(p, m);   // head projection paras

    for (int i = 0; i < para_num_passes; i++) {
        // for each training sample j
        // printf("epoch: %d\n", i);
        for (int j = 0; j < n / b; j++) { // n/b is the total number of blocks.
            // update a(wt), para_b(wt), and alpha(wt)
            a_wt = cblas_ddot(p, wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(p, wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;

            // current learning rate
            eta_t = para_xi / sqrt(t);

            // receive a block of training samples to calculate the gradient
            cblas_dcopy(p, zero_vector, 1, grad_wt, 1);
            for (int kk = 0; kk < b; kk++) {
                const double *cur_xt = x_tr + j * b * p + kk * p;
                double cur_yt = y_tr[j * b + kk];
                double weight;
                if (cur_yt > 0) {
                    weight = 2. * (1.0 - prob_p) * (cblas_ddot(p, wt, 1, cur_xt, 1) - a_wt);
                    weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
                } else {
                    weight = 2.0 * prob_p * (cblas_ddot(p, wt, 1, cur_xt, 1) - b_wt);
                    weight += 2.0 * (1.0 + alpha_wt) * prob_p;
                }
                if (para_verbose > 0) {
                    printf("cur_iter: %05d lr: %.4f a_wt: %.4f b_wt: %.4f alpha_wt: %.4f "
                           "weight: %.4f\n", j, eta_t, a_wt, b_wt, alpha_wt, weight);
                }
                // calculate the gradient
                cblas_daxpy(p, weight, cur_xt, 1, grad_wt, 1);
            }

            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, aver_grad, 1);
            cblas_daxpy(p, 1. / t, grad_wt, 1, aver_grad, 1);

            cblas_dscal(p, 1. / (b * 1.0), grad_wt, 1);



            //gradient descent step: u= wt - eta * grad(wt)
            cblas_dcopy(p, wt, 1, u, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, u, 1);

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
            cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.), u, 1);

            //to do graph projection.
            for (int kk = 0; kk < p; kk++) {
                proj_prizes[kk] = u[kk] * u[kk];
            }
            int g = 1, sparsity_low = para_sparsity, sparsity_high = para_sparsity + 2;
            int tail_max_iter = 50, verbose = 0;
            head_tail_binsearch(edges, weights, proj_prizes, p, m, g, -1, sparsity_low,
                                sparsity_high, tail_max_iter, GWPruning, verbose, graph_stat);
            cblas_dscal(p, 0.0, wt, 1);
            for (int kk = 0; kk < graph_stat->re_nodes->size; kk++) {
                int cur_node = graph_stat->re_nodes->array[kk];
                wt[cur_node] = u[cur_node];
            }


            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, wt_bar, 1);
            cblas_daxpy(p, 1. / t, wt, 1, wt_bar, 1);

            // to calculate AUC score and run time
            if ((fmod(t, para_step_len) == 0.)) {
                double eval_start = clock();
                double *y_pred = malloc(sizeof(double) * n);
                cblas_dgemv(CblasRowMajor, CblasNoTrans,
                            n, p, 1., x_tr, p, wt_bar, 1, 0.0, y_pred, 1);
                double auc = _auc_score(y_tr, y_pred, n);
                free(y_pred);
                double eval_time = (clock() - eval_start) / CLOCKS_PER_SEC;
                results->t_eval_time += eval_time;
                double run_time = (clock() - start_time) / CLOCKS_PER_SEC - results->t_eval_time;
                results->t_run_time[results->t_index] = run_time;
                results->t_auc[results->t_index] = auc;
                results->t_indices[results->t_index] = i * n + j;
                results->t_index++;
                if (para_verbose > 0) {
                    printf("current auc score: %.4f\n", auc);
                }
            }
            // increase time
            t++;
        }
    }
    cblas_dcopy(p, wt_bar, 1, results->wt_bar, 1);
    cblas_dcopy(p, wt, 1, results->wt, 1);
    free(proj_prizes);
    free(proj_costs);
    free_graph_stat(graph_stat);
    free(aver_grad);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
    free(wt_bar);
    free(wt);
    free(zero_vector);
    return true;
}


bool __graph_am_sparse(const double *x_values,// the values of these nonzeros.
                       const int *x_indices,  // the inidices of these nonzeros.
                       const int *x_positions,// the start indices of these nonzeros.
                       const int *x_len_list, // the list of sizes of nonzeros.
                       const double *y_tr,    // the vector of training samples.
                       int p,                 // the dimension of the features of the dataset
                       int n,                 // the total number of training samples.
                       int b,
                       int para_sparsity,
                       double para_xi,
                       double para_l2_reg,
                       int num_passes,
                       int step_len,
                       int verbose,
                       graph_am_results *results) {
    // start time clock
    double start_time = clock();

    // zero vector and set it  to zero.
    double *zero_vector = malloc(sizeof(double) * p);
    memset(zero_vector, 0, sizeof(double) * p);

    // wt --> 0.0
    double *wt = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt, 1);
    // wt_bar --> 0.0
    double *wt_bar = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, wt_bar, 1);
    // gradient
    double *grad_wt = malloc(sizeof(double) * p);
    // proxy vector
    double *u = malloc(sizeof(double) * p);

    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=1]
    double a_wt, *posi_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, posi_x_mean, 1);
    // the estimate of the expectation of positive sample x., i.e. w^T*E[x|y=-1]
    double b_wt, *nega_x_mean = malloc(sizeof(double) * p);
    cblas_dcopy(p, zero_vector, 1, nega_x_mean, 1);
    // initialize alpha_wt (initialize to zero.)
    double alpha_wt;

    // to determine a_wt, b_wt, and alpha_wt
    double posi_t = 0.0, nega_t = 0.0;
    double *full_v = malloc(sizeof(double) * p);
    for (int i = 0; i < n; i++) {
        // get current sample
        const double *cur_xt = x_values + x_positions[i];
        const int *cur_indices = x_indices + x_positions[i];
        _sparse_to_full(cur_xt, cur_indices, x_len_list[i], full_v, p);
        double yt = y_tr[i];
        if (yt > 0) {
            posi_t++;
            cblas_dscal(p, (posi_t - 1.) / posi_t, posi_x_mean, 1);
            cblas_daxpy(p, 1. / posi_t, full_v, 1, posi_x_mean, 1);
        } else {
            nega_t++;
            cblas_dscal(p, (nega_t - 1.) / nega_t, nega_x_mean, 1);
            cblas_daxpy(p, 1. / nega_t, full_v, 1, nega_x_mean, 1);
        }
    }

    // initialize the estimate of probability p=Pr(y=1)
    double prob_p = posi_t / (n * 1.0);

    if (verbose > 0) {
        printf("num_posi: %f num_nega: %f prob_p: %.4f\n", posi_t, nega_t, prob_p);
        printf("average norm(x_posi): %.4f average norm(x_nega): %.4f\n",
               sqrt(cblas_ddot(p, posi_x_mean, 1, posi_x_mean, 1)),
               sqrt(cblas_ddot(p, nega_x_mean, 1, nega_x_mean, 1)));
    }

    // learning rate
    double eta_t;

    // initial start time is zero=1.0
    double t = 1.0;

    //intialize the results
    results->t_index = 0;
    results->t_eval_time = 0.0;

    for (int i = 0; i < num_passes; i++) {
        // for each training sample j
        // printf("epoch: %d\n", i);
        double per_s_time = clock();
        for (int j = 0; j < n; j++) {
            // receive training sample zt=(xt,yt)
            const double *cur_xt = x_values + x_positions[j];
            const int *cur_indices = x_indices + x_positions[j];
            double cur_yt = y_tr[j];

            // current learning rate
            eta_t = para_xi / sqrt(t);

            // update a(wt), para_b(wt), and alpha(wt)
            a_wt = cblas_ddot(p, wt, 1, posi_x_mean, 1);
            b_wt = cblas_ddot(p, wt, 1, nega_x_mean, 1);
            alpha_wt = b_wt - a_wt;

            double weight;
            double dot_prod = _sparse_dot(cur_indices, cur_xt, x_len_list[j], wt);
            if (cur_yt > 0) {
                weight = 2. * (1.0 - prob_p) * (dot_prod - a_wt);
                weight -= 2. * (1.0 + alpha_wt) * (1.0 - prob_p);
            } else {
                weight = 2.0 * prob_p * (dot_prod - b_wt);
                weight += 2.0 * (1.0 + alpha_wt) * prob_p;
            }
            if (verbose > 0) {
                printf("cur_iter: %05d lr: %.4f a_wt: %.4f b_wt: %.4f alpha_wt: %.4f "
                       "weight: %.4f\n", j, eta_t, a_wt, b_wt, alpha_wt, weight);
            }

            memset(grad_wt, 0, sizeof(double) * p);
            for (int kk = 0; kk < x_len_list[j]; kk++) {
                grad_wt[cur_indices[kk]] = weight * cur_xt[kk];
            }
            if (false) {
                // calculate the gradient
                _sparse_to_full(cur_xt, cur_indices, x_len_list[j], full_v, p);
                cblas_dcopy(p, full_v, 1, grad_wt, 1);
                cblas_dscal(p, weight, grad_wt, 1);
            }

            // gradient descent step: u= wt - eta * grad(wt)
            cblas_dcopy(p, wt, 1, u, 1);
            cblas_daxpy(p, -eta_t, grad_wt, 1, u, 1);

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
            cblas_dscal(p, 1. / (eta_t * para_l2_reg + 1.), u, 1);
            _hard_thresholding(u, p, para_sparsity); // k-sparse step.
            cblas_dcopy(p, u, 1, wt, 1);

            // take average of wt --> wt_bar
            cblas_dscal(p, (t - 1.) / t, wt_bar, 1);
            cblas_daxpy(p, 1. / t, wt, 1, wt_bar, 1);

            // to calculate AUC score and run time
            if (fmod(t, step_len) == 0.) {
                printf("test!\n");
                double eval_start = clock();
                double *y_pred = malloc(sizeof(double) * n);
                for (int q = 0; q < n; q++) {
                    cur_xt = x_values + x_positions[q];
                    cur_indices = x_indices + x_positions[q];
                    y_pred[q] = _sparse_dot(cur_indices, cur_xt, x_len_list[q], wt_bar);
                }
                double auc = _auc_score(y_tr, y_pred, n);
                free(y_pred);
                double eval_time = (clock() - eval_start) / CLOCKS_PER_SEC;
                results->t_eval_time += eval_time;
                double run_time = (clock() - start_time) / CLOCKS_PER_SEC - results->t_eval_time;
                results->t_run_time[results->t_index] = run_time;
                results->t_auc[results->t_index] = auc;
                results->t_indices[results->t_index] = i * n + j;
                results->t_index++;
                if (verbose == 0) {
                    printf("current auc score: %.4f\n", auc);
                }
            }
            // increase time
            t++;
        }
        if (verbose == 0) {
            printf("run time: %.4f\n", (clock() - per_s_time) / CLOCKS_PER_SEC);
        }
    }
    cblas_dcopy(p, wt_bar, 1, results->wt_bar, 1);
    cblas_dcopy(p, wt, 1, results->wt, 1);
    free(full_v);
    free(nega_x_mean);
    free(posi_x_mean);
    free(u);
    free(grad_wt);
    free(wt_bar);
    free(wt);
    free(zero_vector);
    return true;
}


void algo_opauc(const double *x_tr,
                const double *y_tr,
                int p,
                int n,
                double eta, double lambda, double *wt, double *wt_bar) {
    double num_p = 0.0, num_n = 0.0;
    double *center_p = malloc(sizeof(double) * p);
    double *center_n = malloc(sizeof(double) * p);
    double *cov_p = malloc(sizeof(double) * p * p);
    double *cov_n = malloc(sizeof(double) * p * p);
    double *grad_wt = malloc(sizeof(double) * p);
    memset(center_p, 0, sizeof(double) * p);
    memset(center_n, 0, sizeof(double) * p);
    memset(cov_p, 0, sizeof(double) * p * p);
    memset(cov_n, 0, sizeof(double) * p * p);
    memset(wt, 0, sizeof(double) * p);
    memset(wt_bar, 0, sizeof(double) * p);

    double *tmp_mat = malloc(sizeof(double) * p * p);
    double *tmp_vec = malloc(sizeof(double) * p);
    memset(tmp_mat, 0, sizeof(double) * p);
    memset(tmp_vec, 0, sizeof(double) * p);


    for (int t = 0; t < n; t++) {
        const double *cur_x = x_tr + t * p;
        double cur_y = y_tr[t];
        if (cur_y > 0) {
            num_p++;
            // copy previous center
            cblas_dcopy(p, center_p, 1, tmp_vec, 1);
            // update center_p
            cblas_dscal(p, (num_p - 1.) / num_p, center_p, 1);
            cblas_daxpy(p, 1. / num_p, cur_x, 1, center_p, 1);
            // update covariance matrix
            cblas_dscal(p * p, (num_p - 1.) / num_p, cov_p, 1);
            cblas_dger(CblasRowMajor, p, p, 1. / num_p, cur_x, 1, cur_x, 1, cov_p, p);
            cblas_dger(CblasRowMajor, p, p, (num_p - 1.) / num_p,
                       tmp_vec, 1, tmp_vec, 1, cov_p, p);
            cblas_dger(CblasRowMajor, p, p, -1., center_p, 1, center_p, 1, cov_p, p);
            if (num_n > 0.0) {
                // calculate the gradient part 1: \lambda w + x_t - c_t^+
                cblas_dcopy(p, center_n, 1, grad_wt, 1);
                cblas_daxpy(p, -1., cur_x, 1, grad_wt, 1);
                cblas_daxpy(p, lambda, wt, 1, grad_wt, 1);
                // xt - c_t^-
                cblas_dcopy(p, cur_x, 1, tmp_vec, 1);
                cblas_daxpy(p, -1., center_n, 1, tmp_vec, 1);
                // (xt - c_t^+)(xt - c_t^+)^T
                cblas_dscal(p * p, 0.0, tmp_mat, 1);
                cblas_dger(CblasRowMajor, p, p, 1., tmp_vec, 1, tmp_vec, 1, tmp_mat, p);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, p, p, 1., tmp_mat, p, wt, 1, 1.0, grad_wt,
                            1);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, p, p, 1., cov_n, p, wt, 1, 1.0, grad_wt,
                            1);
            } else {
                cblas_dscal(p, 0.0, grad_wt, 1);
            }
        } else {
            num_n++;
            // copy previous center
            cblas_dcopy(p, center_n, 1, tmp_vec, 1);
            // update center_n
            cblas_dscal(p, (num_n - 1.) / num_n, center_n, 1);
            cblas_daxpy(p, 1. / num_n, cur_x, 1, center_n, 1);
            // update covariance matrix
            cblas_dscal(p * p, (num_n - 1.) / num_n, cov_n, 1);
            cblas_dger(CblasRowMajor, p, p, 1. / num_n, cur_x, 1, cur_x, 1, cov_n, p);
            cblas_dger(CblasRowMajor, p, p, (num_n - 1.) / num_n,
                       tmp_vec, 1, tmp_vec, 1, cov_n, p);
            cblas_dger(CblasRowMajor, p, p, -1., center_n, 1, center_n, 1, cov_n, p);
            if (num_p > 0.0) {
                // calculate the gradient part 1: \lambda w + x_t - c_t^+
                cblas_dcopy(p, cur_x, 1, grad_wt, 1);
                cblas_daxpy(p, -1., center_p, 1, grad_wt, 1);
                cblas_daxpy(p, lambda, wt, 1, grad_wt, 1);
                // xt - c_t^+
                cblas_dcopy(p, cur_x, 1, tmp_vec, 1);
                cblas_daxpy(p, -1., center_p, 1, tmp_vec, 1);
                // (xt - c_t^+)(xt - c_t^+)^T
                cblas_dscal(p * p, 0.0, tmp_mat, 1);
                cblas_dger(CblasRowMajor, p, p, 1., tmp_vec, 1, tmp_vec, 1, tmp_mat, p);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, p, p, 1., tmp_mat, p, wt, 1, 1.0, grad_wt,
                            1);
                cblas_dgemv(CblasRowMajor, CblasNoTrans, p, p, 1., cov_p, p, wt, 1, 1.0, grad_wt,
                            1);
            } else {
                cblas_dscal(p, 0.0, grad_wt, 1);
            }
        }
        // update the solution
        cblas_daxpy(p, -eta, grad_wt, 1, wt, 1);
        cblas_dscal(p, (t * 1.) / (t * 1. + 1.), wt_bar, 1);
        cblas_daxpy(p, 1. / (t + 1.), wt, 1, wt_bar, 1);
        if (false) {
            printf("||wt||: %.4f ||wt-bar||: %.4f grad: %.4f eta: %.4f "
                   "lambda: %.4f num_p: %.4f num_n: %.4f\n",
                   cblas_ddot(p, wt, 1, wt, 1),
                   cblas_ddot(p, wt_bar, 1, wt_bar, 1),
                   cblas_ddot(p, grad_wt, 1, grad_wt, 1), eta, lambda, num_p, num_n);
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

void project_onto_l1(const double *w, int p, double r, double *proj_v) {

    // if the ||w||_1 <= r, return w directly.
    double w_l1 = 0.0;
    double *abs_w = malloc(sizeof(double) * p);
    for (int i = 0; i < p; i++) {
        abs_w[i] = fabs(w[i]);
        w_l1 += fabs(w[i]);
    }
    if (w_l1 <= r) {
        cblas_dcopy(p, w, 1, proj_v, 1);
        free(abs_w);
        return;
    }
    // sort vector abs_w
    int *sorted_indices = malloc(sizeof(int) * p);
    double *u = malloc(sizeof(double) * p);
    double *sv = malloc(sizeof(double) * p);
    _arg_sort_descend(abs_w, sorted_indices, p);
    double prev_sum = 0.0;
    for (int i = 0; i < p; i++) {
        u[i] = abs_w[sorted_indices[i]];
        sv[i] = prev_sum + u[i];
        prev_sum = sv[i];
    }
    int last_rho = 0;
    for (int i = 0; i < p; i++) {
        if (u[i] > (sv[i] - r) / (i + 1.)) {
            last_rho = i;
        }
    }
    double theta = fmax(0.0, (sv[last_rho] - r) / (last_rho + 1.));
    for (int i = 0; i < p; i++) {
        proj_v[i] = sign(w[i]) * fmax(fabs(w[i]) - theta, 0.0);
    }
    free(sorted_indices);
    free(u);
    free(sv);
    free(abs_w);
}

void algo_fsauc(const double *x_tr, const double *y_tr, int p, int n,
                double para_r, double para_g, int num_passes, double *wt, double *wt_bar) {
    double kappa = 1.0; //assume that kappa=1.0 for normalized data samples.
    double global_n = n * num_passes;

    // all initialize parameters
    int m = (int) floor(0.5 * log2((2.0 * global_n) / log2(global_n))) - 1;
    int n0 = (int) floor(global_n / (double) m);
    double R0 = 2. * sqrt(1. + 2. * pow(kappa, 2.)) * para_r;
    double beta_prev = 1. + 8.0 * pow(kappa, 2.);
    double *v_bar = malloc(sizeof(double) * (p + 2));
    memset(v_bar, 0, sizeof(double) * (p + 2));
    memset(wt, 0, sizeof(double) * p);
    memset(wt_bar, 0, sizeof(double) * p);

    double Rk = R0;


    double *vec_v1 = malloc(sizeof(double) * (p + 2));

    double *vec_w = malloc(sizeof(double) * p);
    double *vec_w1 = malloc(sizeof(double) * p);
    double *w_bar = malloc(sizeof(double) * p);


    double R = para_r;
    double *gw = malloc(sizeof(double) * p);
    double *vec_temp = malloc(sizeof(double) * p);

    int K = 1; // number of alternative projections.



    //some global updates
    // v_bar, p_hat, A, A_p, A_n, n_p, n_n
    double p_hat = 0.0;
    double *vec_a = malloc(sizeof(double) * (p + 2));
    memset(vec_a, 0, sizeof(double) * (p + 2));
    double *vec_a_p = malloc(sizeof(double) * p), n_p = 0.0;
    memset(vec_a_p, 0, sizeof(double) * p);
    double *vec_a_n = malloc(sizeof(double) * p), n_n = 0.0;
    memset(vec_a_n, 0, sizeof(double) * p);

    double gamma = para_g;
    for (int k = 0; k < m; k++) {

        cblas_dcopy(p + 2, v_bar, 1, vec_v1, 1);
        // some parameters for each stage
        double delta = 0.1;
        double alpha1 = cblas_ddot(p + 2, vec_a, 1, vec_v1, 1);
        double alpha = alpha1;
        cblas_dcopy(p, vec_v1, 1, vec_w1, 1);
        cblas_dcopy(p, vec_w1, 1, vec_w, 1);
        double a1 = vec_v1[p];
        double b1 = vec_v1[p + 1];
        double a = a1;
        double b = b1;
        memset(w_bar, 0, sizeof(double) * p);
        double a_bar = 0.0;
        double b_bar = 0.0;
        double alpha_bar = 0.0;

        double D0 = 2.0 * sqrt(2.0) * kappa * R0;
        if (k != 0) {
            // not in the first stage

            D0 += (4.0 * sqrt(2.0) * kappa * (2.0 + sqrt(2.0 * log(12.0 / delta))) *
                   (1.0 + 2.0 * kappa) * R) /
                  sqrt(min(p_hat, 1.0 - p_hat) * (double) n0 - sqrt(2.0 * n0 * log(12.0 / delta)));
            if (D0 <= 0.0) {
                printf("D0 is negative!");
                free(vec_a_n);
                free(vec_a_p);
                free(vec_a);
                free(gw);
                free(vec_temp);
                free(vec_v1);
                free(vec_w);
                free(vec_w1);
                free(v_bar);
                exit(0);
            }
        }

        double n_posi_nega = n_p + n_n;
        for (int kk = 0; kk < n0; kk++) {
            int global_index = k * n0 + kk;
            const double *xt = x_tr + (global_index % n) * p;
            double yt = y_tr[global_index % n];
            double is_posi = (yt == 1.0 ? 1. : 0.0);
            double is_nega = (yt == -1.0 ? 1. : 0.0);
            p_hat = ((n_posi_nega + kk) * p_hat + is_posi) / (kk + 1. + n_posi_nega);

            // this is update a_p, a_n, n_p, n_n
            cblas_daxpy(p, is_posi, xt, 1, vec_a_p, 1); // 89: A_p = A_p + (X*(y==1))';
            cblas_daxpy(p, is_nega, xt, 1, vec_a_n, 1); // 90: A_n = A_n + (X*(y==-1))';
            n_p += is_posi; // 91: n_p = n_p + sum(y==1);
            n_n += is_nega; // 92: n_n = n_n + sum(y==-1);

            // compute gradient
            double pred = cblas_ddot(p, vec_w, 1, xt, 1);
            double temp = 2. * (is_nega * p_hat - is_posi * (1. - p_hat));
            double ga = 2. * is_posi * (1. - p_hat) * (a - pred);
            double gb = 2. * is_nega * p_hat * (b - pred);
            memset(gw, 0, sizeof(double) * p);
            cblas_daxpy(p, (1. + alpha) * temp - ga - gb, xt, 1, gw, 1);
            double galpha = temp * pred - 2.0 * p_hat * (1. - p_hat) * alpha;

            // updates w,a,b,alpha
            cblas_daxpy(p, -gamma, gw, 1, vec_w, 1); // w = w-gamma*gw;
            a += -gamma * ga; // a = a-gamma*ga;
            b += -gamma * gb; // b = b-gamma*gb;
            alpha += gamma * galpha;

            // project: w,a,b
            for (int j = 0; j < K; j++) {
                project_onto_l1(vec_w, p, R, vec_temp); //project w on L1 ball.
                cblas_dcopy(p, vec_temp, 1, vec_w, 1);
                a = sign(a) * fmin(fabs(a), R * kappa); //project a
                b = sign(b) * fmin(fabs(b), R * kappa); //project b
                // 65:72
                cblas_dcopy(p, vec_w1, 1, vec_temp, 1);
                cblas_daxpy(p, -1., vec_w, 1, vec_temp, 1);
                double v_norm = cblas_ddot(p, vec_temp, 1, vec_temp, 1);
                v_norm += (a - a1) * (a - a1) + (b - b1) * (b - b1);
                v_norm = sqrt(v_norm);
                if (v_norm > R0) {
                    double tmp = R0 / v_norm;
                    cblas_dscal(p, tmp, vec_w, 1);
                    cblas_daxpy(p, 1., vec_w1, 1, vec_temp, 1);
                    a = a1 + tmp * a;
                    b = b1 + tmp * b;
                }
            }
            //project: alpha
            double tmp = fmin(fmin(D0 + alpha1, 2. * R * kappa), alpha);
            alpha = fmax(-D0 + alpha1, fmax(-2 * R * kappa, tmp));

            // 80:86
            cblas_dscal(p, 1. / (kk + 1.), w_bar, 1);
            cblas_daxpy(p, 1. / (kk + 1.), vec_w, 1, w_bar, 1);
            a_bar = (kk * a_bar + a) / (kk + 1.);
            b_bar = (kk * b_bar + b) / (kk + 1.);
            alpha_bar = (kk * alpha_bar + alpha) / (kk + 1.);
        }//inner-stage
        Rk /= 2.0;


        cblas_dcopy(p, vec_a_n, 1, vec_a, 1); // 93: A = [A_n/n_n - A_p/n_p,0,0];
        cblas_dscal(p, 1. / n_n, vec_a, 1);
        cblas_daxpy(p, -1. / n_p, vec_a_p, 1, vec_a, 1);
        vec_a[p] = 0.0;
        vec_a[p + 1] = 0.0;

        cblas_dcopy(p, w_bar, 1, v_bar, 1); // 94: v_bar = [w_bar;a_bar;b_bar];
        v_bar[p] = a_bar;
        v_bar[p + 1] = b_bar;
        // update beta and D
        double tmp1 = pow(kappa * (1. + 2. * kappa) * (2. + sqrt(2. * log(12. / delta))), 2.);
        double tmp2 = fmin(p_hat, 1. - p_hat) - sqrt(2. * log(12. / delta) / n0);
        double beta_next = (1. + 8.0 * pow(kappa, 2.)) + (32. * tmp1) / tmp2; //update beta

        if (beta_next <= 0.0) {
            printf("beta_next is negative!");
            free(vec_a_n);
            free(vec_a_p);
            free(vec_a);
            free(gw);
            free(vec_temp);
            free(vec_v1);
            free(vec_w);
            free(vec_w1);
            free(v_bar);
            exit(0);
        }

        cblas_dcopy(p, v_bar, 1, wt, 1);
        cblas_dscal(p, 1. / (k + 1.), wt_bar, 1);
        cblas_daxpy(p, 1. / (k + 1.), wt, 1, wt_bar, 1);

        // to make sure the step size is not increasing.
        gamma = fmin(sqrt(beta_next / beta_prev) * (gamma / 2.), gamma);
        beta_prev = beta_next;
    }//outer-stage

    free(vec_a_n);
    free(vec_a_p);
    free(vec_a);
    free(gw);
    free(vec_temp);
    free(vec_v1);
    free(vec_w);
    free(vec_w1);
    free(v_bar);
}
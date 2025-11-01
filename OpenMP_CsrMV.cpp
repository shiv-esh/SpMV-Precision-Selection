1 void OmpMergeCsrmv(int num_threads, const CsrMatrix& A, double* x, double* y) 
2 { 
3     int* row_end_offsets = A.row_offsets + 1;                                  // Merge list A: row end-offsets 
4     CountingInputIterator<int> nz_indices(0);                                  // Merge list B: Natural numbers(NZ indices) 
5     int num_merge_items = A.num_rows + A.num_nonzeros;                         // Merge path total length 
6     int items_per_thread = (num_merge_items + num_threads - 1) / num_threads;  // Merge items per thread 
7  
8     int row_carry_out[num_threads]; 
9     double value_carry_out[num_threads]; 
10  
11     // Spawn parallel threads 
12     #pragma omp parallel for schedule(static) num_threads(num_threads) 
13     for (int tid = 0; tid < num_threads; tid++) 
14     { 
15         // Find starting and ending MergePath coordinates (row-idx, nonzero-idx) for each thread 
16         int diagonal                 = min(items_per_thread * tid, num_merge_items); 
17         int diagonal_end             = min(diagonal + items_per_thread, num_merge_items); 
18         CoordinateT thread_coord     = MergePathSearch(diagonal, row_end_offsets, nz_indices, 
19                                            A.num_rows, A.num_nonzeros); 
20         CoordinateT thread_coord_end = MergePathSearch(diagonal_end, row_end_offsets, nz_indices, 
21                                            A.num_rows, A.num_nonzeros); 
22  
23         // Consume merge items, whole rows first 
24         double running_total = 0.0; 
25         for (; thread_coord.x < thread_coord_end.x; ++thread_coord.x) 
26         { 
27             for (; thread_coord.y < row_end_offsets[thread_coord.x]; ++thread_coord.y) 
28                 running_total += A.values[thread_coord.y] * x[A.column_indices[thread_coord.y]]; 
29  
30             y[thread_coord.x] = running_total; 
31             running_total = 0.0; 
32         } 
33  
34         // Consume partial portion of thread's last row 
35         for (; thread_coord.y < thread_coord_end.y; ++thread_coord.y) 
36             running_total += A.values[thread_coord.y] * x[A.column_indices[thread_coord.y]]; 
37  
38         // Save carry-outs 
39         row_carry_out[tid] = thread_coord_end.x; 
40         value_carry_out[tid] = running_total; 
41     } 
42  
43     // Carry-out fix-up (rows spanning multiple threads) 
44     for (int tid = 0; tid < num_threads - 1; ++tid) 
45         if (row_carry_out[tid] < A.num_rows) 
46             y[row_carry_out[tid]] += value_carry_out[tid]; 
47 } 
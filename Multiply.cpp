#include<iostream>
#include <vector>
#include<stdlib.h>
#include<mpi.h>
#include<time.h>
#include <fstream>

using namespace std;

int main(int argc, char* argv[])
{
    srand(time(NULL));
    const char* path_1 = "/home/2021-03959/matrix1.txt";
    const char* path_2 = "/home/2021-03959/matrix2.txt";
    const char* res = "/home/2021-03959/res.txt";
    const char* stats = "/home/2021-03959/stats.txt";

    double start, stop;
    int rows;
    int N = 10;
    int rank, numprocs;

    if (argc > 1) {
        path_1 = argv[1];
        path_2 = argv[2];
        res = argv[3];
        N = atoi(argv[4]);
        stats = argv[5];
    }
    else
    {
        cout << "error" << endl;
        return -1;
    }

    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    rows = N / numprocs;
    int* matrixA = (int*)malloc(sizeof(int) * N * N);
    int* matrixB = (int*)malloc(sizeof(int) * N * N);
    int* matrixC = (int*)malloc(sizeof(int) * N * N);
    int* buffer = (int*)malloc(sizeof(int) * N * rows);
    int* result = (int*)malloc(sizeof(int) * N * N);

    if (rank == 0)
    {
        start = MPI_Wtime();

        ifstream fin;
        fin.open(path_1);
        if (fin.is_open())
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    fin >> matrixA[i * N + j];
                }
            }
        }
        fin.close();

        fin.open(path_2);
        if (fin.is_open())
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    fin >> matrixB[i * N + j];
                }
            }
        }
        fin.close();

        for (int i = 1; i < numprocs; i++)
            MPI_Send(matrixB, N * N, MPI_INT, i, 0, MPI_COMM_WORLD);
        for (int i = 1; i < numprocs; i++)
            MPI_Send(matrixA + (i - 1) * rows * N, N * N, MPI_INT, i, 1, MPI_COMM_WORLD);
        for (int k = 1; k < numprocs; k++) {
            MPI_Recv(matrixC, rows * N, MPI_INT, k, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (int i = 0; i < N; i++)
                for (int j = 0; j < N; j++)
                    result[((k - 1) * N + i) * N + j] = matrixC[i * N + j];
        }
        for (int i = (numprocs - 1) * rows; i < N; i++) {
            for (int j = 0; j < N; j++) {
                result[i * N + j] = 0;
                for (int k = 0; k < N; k++)
                    result[i * N + j] += matrixA[i * N + k] * matrixB[k * N + j];
            }
        }

        ofstream fout;
        fout.open(res, ios::app);
        double time = double(clock() - start / CLOCKS_PER_SEC);
        if (fout.is_open())
        {
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    fout << result[i * N + j] << " ";
                }
                fout << "\n";
            }
        }
        fout.close();

        cout << N * N << " " << time << endl;

        stop = MPI_Wtime();

        free(matrixA);
        free(matrixB);
        free(matrixC);
        free(buffer);
        free(result);
    }
    else
    {
        MPI_Recv(matrixB, N * N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(buffer, rows * N, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int i = 0; i < rows; ++i)
        {
            for (int j = 0; j < N; ++j) {
                matrixC[i * N + j] = 0;
                for (int k = 0; k < N; k++)
                    matrixC[i * N + j] += (buffer[i * N + k] * matrixB[k * N + j]);
            }
        }
        MPI_Send(matrixC, rows * N, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
    MPI_Finalize();

    return 0;
}
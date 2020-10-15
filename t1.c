/*Grupo 2 - Turma B
João Vítor Vasconcelos Ponte  10273971
Leonardo Moreira Kobe          9778623
Ricardo Alves de Araujo        9364890
Tiago Esperança Triques        9037713
Yan Crisóstomo Rohwedder       9779263
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define T 8

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Wrong arguments.\n");
        exit(1);
    }

    int num_threads;
    long int maior = -1;
    long int localmaior = -1;
    long int tamanho_vetor = atoi(argv[1]);

    long int *vetor = NULL;
    vetor = (long int*) malloc(tamanho_vetor * sizeof(long int));

    double wtime = omp_get_wtime();

    #pragma omp parallel num_threads(T) shared(maior) private(localmaior)
    {
        int my_id;

        //determina o número da thread
        my_id = omp_get_thread_num();

        //determina o número de threads
        num_threads = omp_get_num_threads();

        //cálculo dos limites das iterações do loop
        long int init = my_id * (tamanho_vetor / num_threads);
        long int final =  (my_id + 1) * (tamanho_vetor / num_threads);

        //preenche o vetor
        for (long int i = init; i < final; i++){
            vetor[i] = 1;
            if (i == tamanho_vetor/2)
                vetor[i] = tamanho_vetor;
        }

        //sincrozina as threads
        #pragma omp barrier

        //determina localmaior
        localmaior = -1;
        for (long int i = init; i < final; i++){
            if (vetor[i] > localmaior){
                localmaior = vetor[i];
            }
        }

        //determina maior
        #pragma omp critical
        if(localmaior > maior)
            maior = localmaior;
    }

    wtime = omp_get_wtime() - wtime;

    printf("maior = %ld, time = %.5f\n", maior, wtime);

    free(vetor);

    return 0;
}

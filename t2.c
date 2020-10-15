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
#include <math.h>

#define T 8

typedef struct metrics {
	double arith_mean;
	double harm_mean;
	double median;
	double mode;
	double variance;
	double std_dev;
	double coef;
} Metrics;

int cmp(const void *a, const void *b){
	return *(double *)a - *(double *)b;
}

double calculate_mode(double *v, int n){
	double cur_mode = 0.0, cur = 0.0;
	int mode_count = 0, cur_count = 0;
	for(int i = 0, cur = v[i]; i < n; i++) {
		if(cur == v[i])	cur_count++;
		else {
			cur = v[i];
			cur_count = 1;
		}
		if(mode_count < cur_count) {
			mode_count = cur_count;
			cur_mode = cur;
		}
	}
	return mode_count == 1 ? -1.0 : cur_mode;
};

double calculate_median(double *v, int n){
	qsort(v, n, sizeof(double), cmp);
	return !(n%2) ? (v[(n/2)-1]+v[(n/2)])/2 : v[n/2];
};

double calculate_arithmetic_mean(double *v, int n) {
	double sum = 0.0;
	
	for(int i = 0; i < n; i++)
		sum += v[i];

	return sum/(double)n;
}

double calculate_harmonic_mean(double *v, int n) {
	double sum = 0.0;
	
	for(int i = 0; i < n; i++)
		sum += 1.0/v[i];

	return (double)n/sum;
}

double calculate_variance(double *v, int n, double avg) {
	double sum = 0.0;

	for(int i = 0; i < n; i++)
		sum += pow(v[i] - avg, 2);

	return sum/(double)(n-1);
}

int main(int argc, char *argv[]) {
	int a;
	int i, j;
    double **matrix = NULL;
	int rows, cols;
	rows = cols = 0;

	a=scanf("%d %d", &rows, &cols);

	// Recebendo a matriz e já armazenando a transposta
	// para facilitar depois :)
	matrix = (double **) calloc(cols, sizeof(double *));
	for(i = 0; i < cols; matrix[i++] = (double *) calloc(rows, sizeof(double)));
	for(i = 0, j = 0; i < rows; i++)
		for(j = 0; j < cols; a=scanf("%lf", &matrix[j++][i]));

	// Como estamos trabalhando com a transposta
	// fazemos o swap de rows e cols
	rows ^= cols; cols ^= rows; rows ^= cols;

	//Vetor de struct para armazenar as métricas
	Metrics *metrics = (Metrics *) calloc(rows, sizeof(Metrics));

	int x[rows];

	double wtime = omp_get_wtime();

    #pragma omp parallel num_threads(8) private(i) shared(metrics, matrix)
    {
        #pragma omp single
        {
            for(i = 0; i < rows; i++) {
                #pragma omp task depend(out: x[i])
                metrics[i].median = calculate_median(matrix[i], cols);
                
				#pragma omp task depend(in: x[i])
                {
                    metrics[i].arith_mean = calculate_arithmetic_mean(matrix[i], cols);
                    metrics[i].variance = calculate_variance(matrix[i], cols, metrics[i].arith_mean);
                    metrics[i].std_dev = sqrt(metrics[i].variance);
                    metrics[i].coef  = metrics[i].std_dev/metrics[i].arith_mean;
                }

                #pragma omp task depend(in: x[i])
                metrics[i].harm_mean = calculate_harmonic_mean(matrix[i], cols);
                
                #pragma omp task depend(in: x[i])
				metrics[i].mode = calculate_mode(matrix[i], cols);
            }
        }
    }

	wtime = omp_get_wtime() - wtime;

	//Impressão das métricas
	for(i = 0; i < rows; i++) {
		printf("%.1lf ", metrics[i].arith_mean);
	}
	printf("\n");
	for(i = 0; i < rows; i++) {
		printf("%.1lf ", metrics[i].harm_mean);
	}
	printf("\n");
	for(i = 0; i < rows; i++) {
		printf("%.1lf ", metrics[i].median);
	}
	printf("\n");
	for(i = 0; i < rows; i++) {
		printf("%.1lf ", metrics[i].mode);
	}
	printf("\n");
	for(i = 0; i < rows; i++) {
		printf("%.1lf ", metrics[i].variance);
	}
	printf("\n");
	for(i = 0; i < rows; i++) {
		printf("%.1lf ", metrics[i].std_dev);
	}
	printf("\n");
	for(i = 0; i < rows; i++) {
		printf("%.1lf ", metrics[i].coef);
	}
	printf("\n");

	fprintf(stderr, "%.3lf\n", wtime);

	for(i = 0; i < rows; i++) free(matrix[i]);
	free(matrix);
	free(metrics);

    return 0;
}

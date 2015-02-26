/**
Program to find the minimal energy in the Lieb-Liniger model as a function of the interaction parameter c in the framework of the Continuous Matrix Product State (cMPS)
Based on the code by M. Rispler
Jorge Alda
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <string.h>

//GNU Scientific Library headers
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>//Random number generator
#include <gsl/gsl_vector.h> //Vector handling
#include <gsl/gsl_matrix.h> //Matrix handling
#include <gsl/gsl_complex.h> //Complex numbers
#include <gsl/gsl_complex_math.h> //Complex number operations
#include <gsl/gsl_blas.h> //Linear algebra engine
#include <gsl/gsl_linalg.h>

//Optimization
#include <nlopt.h>

//#define Npart 1.0 //Expectation value of particle number

void extract(const gsl_vector *x, gsl_matrix_complex *H0, gsl_matrix_complex *R0, unsigned int D);
void enlarge(double *orig, double *dest, unsigned int n, gsl_rng *rng);
void spre(gsl_matrix_complex *A,  gsl_matrix_complex *preA, unsigned int D);
void reprM(gsl_matrix_complex *R0, unsigned int dim);
void spost(const gsl_matrix_complex *A, gsl_matrix_complex *postA, unsigned int trans, unsigned int D);
void steadystatefinder(gsl_matrix_complex *H, gsl_matrix_complex *R, gsl_matrix_complex *rho, unsigned int D);
double matrixnorm(gsl_matrix_complex *A);
void powermethod (gsl_matrix_complex *A, gsl_vector_complex *v, unsigned int D);
int gsl_linalg_complex_QR_decomp (gsl_matrix_complex * A);
double particlenumber(gsl_matrix_complex *R, gsl_matrix_complex *rho, unsigned int D);
double energydensity(gsl_matrix_complex *H, gsl_matrix_complex *R, gsl_matrix_complex *rho, double c, unsigned int D);
void save(double *x, char *t, unsigned int D);

double barrier(unsigned int n, const double *x, const double *grad, void *myfuncdata);
double enr, np, eps=1e-6, Npart=0.53;

int main(){
    int D, j, k, Nbuc;
    double min, max, delta;
    char input[200], output[500];
    FILE *settings;
    if((settings=fopen("settings.txt", "rt"))==NULL){
        printf("no existe archivo settings");
        return -1;
    }
    fscanf(settings, "%s", input);
    fscanf(settings, "%s", output);
    fscanf(settings, "%d", &D);
    fscanf(settings, "%lf", &min);
    fscanf(settings, "%lf", &max);
    fscanf(settings, "%lf", &delta);
    fscanf(settings, "%d", &Nbuc);
    double x[2*D*D];
    FILE *f;
    time_t start, stop;


    if (strcmp(input, "r")==0){
        printf("random\n");
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
    for(j=0; j<2*(D)*(D); j++){
       x[j] = gsl_rng_uniform(rng);
    }
    }else{
    f = fopen(input, "rt");
    for(j=0; j<2*(D)*(D); j++){
       fscanf(f, "%lf", &x[j]);
    }
    fclose(f);
    printf("leido el archivo %s", input);
    }

    double c[2], mm;
    c[1] = D;
    c[0] = min;


    nlopt_opt opt = nlopt_create(NLOPT_LN_SBPLX, 2*D*D);
    nlopt_set_min_objective(opt, barrier, c);
    nlopt_set_xtol_rel(opt, 1e-4);
    nlopt_set_maxeval(opt, 30000);

    //Find the groundstate
    sprintf(arch, "LL_%d.txt", D);
    time(&start);
    for(j=0; j<Nbuc; j++){
    for(c[0] = min; c[0]<max; c[0]+= delta){
        nlopt_optimize(opt, x, &mm);
        f = fopen(arch, "at");
        fprintf(f, "%lf\t%lf\n", c[0], enr);
        fclose(f);
    }
    printf("%lf\n", enr);
    for(; c[0]>min; c[0]-= delta){
        nlopt_optimize(opt, x, &mm);
        f = fopen(arch, "at");
        fprintf(f, "%lf\t%lf\n", c[0], enr);
        fclose(f);
    }
    }
    time(&stop):
    save(x, output, D);
    printf("%0.f s\n", difftime(stop, start));
    return 0;
}


double barrier(unsigned int n, const double *x, const double *grad, void *myfuncdata){
    //Barrier function to be minimized
    double e, N, L;
    double *c = (double *)myfuncdata;
    unsigned int D = (unsigned int)c[1];
    gsl_vector_view v = gsl_vector_view_array(x, 2*D*D);
    gsl_matrix_complex *H = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex *R = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex *rho = gsl_matrix_complex_alloc(D, D);
    extract(&v.vector, H, R, D);
    steadystatefinder(H, R, rho, D);
    e = energydensity(H, R, rho, c[0], D);
    enr = e;
    N = particlenumber(R, rho, D);
    np = N;
    gsl_matrix_complex_free(H);
    gsl_matrix_complex_free(R);
    gsl_matrix_complex_free(rho);
    L = e - 0.0001*log((N-Npart)); //We're using a logarithmic barrier
    return L;
}

void reprM(gsl_matrix_complex *R0, unsigned int dim){
    // Auxiliar function to print matrices
    unsigned int m, n;
    for(m=0; m<dim; m++){
        for(n=0; n<dim; n++){
            printf("%lf+%lfi\t", GSL_REAL(gsl_matrix_complex_get(R0, m, n)), GSL_IMAG(gsl_matrix_complex_get(R0, m, n)));
        }
        printf( "\n");
    }
    printf( "\n");
}

void reprv(gsl_vector_complex *v){
    unsigned int D = v->size, i;
    for(i=0; i<D; i++){
        printf("%.3lf+%.3lfi\t", GSL_REAL(gsl_vector_complex_get(v, i)), GSL_IMAG(gsl_vector_complex_get(v, i)));
    }
    printf("\n");
}

void save(double *x, char *t, unsigned int D){
    int i, j;
    char t1[500], t2[500];
    sprintf(t1, "KR_%s.txt", t);
    sprintf(t2, "rho_%s.txt", t);
    FILE *f = fopen(t1, "wt");
    for(i=0; i<2*D*D; i++){
        fprintf(f, "%lf\n", x[i]);
    }
    fclose(f);
    gsl_vector_view v = gsl_vector_view_array(x, 2*D*D);
    gsl_matrix_complex *H = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex *R = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex *rho = gsl_matrix_complex_alloc(D, D);
    extract(&v.vector, H, R, D);
    steadystatefinder(H, R, rho, D);
    f = fopen(t2, "wt");
    for(j=0; j<D; j++){
        for(i=0; i<D; i++){
            fprintf(f, "%lf\t%lf i\n", GSL_REAL(gsl_matrix_complex_get(rho, i, j)), GSL_IMAG(gsl_matrix_complex_get(rho, i, j)));
        }
    }
    fclose(f);
    gsl_matrix_complex_free(H);
    gsl_matrix_complex_free(R);
    gsl_matrix_complex_free(rho);
}


void extract(const gsl_vector *x, gsl_matrix_complex *H0, gsl_matrix_complex *R0, unsigned int D){
    unsigned int a,n,m;
    for(m=0; m<D;m++){
        gsl_matrix_complex_set(H0, m, m, gsl_complex_rect(gsl_vector_get(x,m),0));
    }
    a = D;
    for(m=0; m<D;m++){
        for(n=m+1; n<D;n++){
            gsl_matrix_complex_set(H0, m, n, gsl_complex_rect(gsl_vector_get(x,a), gsl_vector_get(x,a+1)));
            gsl_matrix_complex_set(H0, n, m, gsl_complex_rect(gsl_vector_get(x,a), -gsl_vector_get(x,a+1)));
            a += 2;
        }
    }

    a=0;
    for(m=0;m<D;m++){
        for(n=0;n<D;n++){
            gsl_matrix_complex_set(R0, n, m, gsl_complex_rect(gsl_vector_get(x,a+D*D),0));
            a++;
        }
    }
}

void enlarge(double *orig, double *dest, unsigned int n, gsl_rng *rng){
    unsigned int i, j, k, l;
    double alfa =0;
    for(i=0; i<n; i++){
        dest[i] = orig[i];
    }
    l=i;
    dest[i] = alfa*gsl_rng_uniform(rng);
    i++;
    for(j=0; j<n-1; j++){
        for(k=0; k<2*(n-j-1); k++){
            i++;
            l++;
            dest[l] = orig[i-2];

        }
        dest[l+2] = alfa*gsl_rng_uniform(rng);
        dest[l+1] = alfa*gsl_rng_uniform(rng);
        l+=2;
    }
    dest[l+2] = alfa*gsl_rng_uniform(rng);
    dest[l+1] = alfa*gsl_rng_uniform(rng);

    for(i=0; i<(n+1)*(n+1); i++){
        j = i/(n+1);
        k = i%(n+1);
        if((j<n)&&(k<n)){
            dest[i+(n+1)*(n+1)] = orig[n*j+k+n*n];
        }else{
            dest[i+(n+1)*(n+1)] = alfa*gsl_rng_uniform(rng);
        }
    }

}

void spre(gsl_matrix_complex *A,  gsl_matrix_complex *preA, unsigned int D){
    gsl_matrix_complex_set_zero(preA);
    unsigned int i;
    for(i=0; i<D;i++){
        gsl_matrix_complex_view block = gsl_matrix_complex_submatrix(preA,i*D,i*D,D,D);
        gsl_matrix_complex_memcpy(&block.matrix,A);
    }
}

void spost(const gsl_matrix_complex *A, gsl_matrix_complex *postA, unsigned int trans, unsigned int D){
    gsl_matrix_complex_set_zero(postA);
    unsigned int i, n, m;
    gsl_complex d;
    for(m=0;m<D;m++){
        for(n=0;n<D;n++){
            if(trans){
                d = gsl_complex_conjugate(gsl_matrix_complex_get(A, n, m));
            }else{
                d = gsl_matrix_complex_get(A, m, n);
            }
            for(i=0;i<D;i++){
                gsl_matrix_complex_set(postA,i+D*n, i+D*m, d);
            }
        }
    }
}

void steadystatefinder(gsl_matrix_complex *H, gsl_matrix_complex *R, gsl_matrix_complex *rho, unsigned int D){
    int i=0, m, n;
    gsl_complex tr = GSL_COMPLEX_ZERO;
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_taus);
    gsl_matrix_complex *L = gsl_matrix_complex_alloc(D*D, D*D);
    gsl_matrix_complex *post = gsl_matrix_complex_alloc(D*D, D*D);
    gsl_matrix_complex *pre = gsl_matrix_complex_alloc(D*D, D*D);
    gsl_matrix_complex *RtR = gsl_matrix_complex_alloc(D, D);
    spre(H,L, D);
    spost(H, post, 0, D);
    gsl_matrix_complex_sub(L,post);
    spre(R, pre, D);
    spost(R, post, 1, D);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(1,0), pre,post, gsl_complex_rect(0, -1), L);
    gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, gsl_complex_rect(1,0), R, R, gsl_complex_rect(0,0), RtR);
    spre(RtR, pre, D);
    spost(RtR, post, 0, D);
    gsl_matrix_complex_add(pre, post);
    gsl_matrix_complex_scale(pre, gsl_complex_rect(0.5,0));
    gsl_matrix_complex_sub(L, pre);

    gsl_matrix_complex_free(pre);
    gsl_matrix_complex_free(RtR);
    gsl_matrix_complex_free(post);

   gsl_vector_complex *vectr;
    vectr = gsl_vector_complex_alloc(D*D);
    for(i=0; i<vectr->size; i++){
        gsl_vector_complex_set(vectr, i, gsl_complex_rect(gsl_rng_uniform(rng),gsl_rng_uniform(rng)));
    }
    gsl_vector_complex_set(vectr, vectr->size-1, GSL_COMPLEX_ZERO);
    powermethod(L, vectr, D*D);
    gsl_rng_free(rng);

    i=0;
    for(m=0; m<D; m++){
        for(n=0; n<D; n++){
            gsl_matrix_complex_set(rho, n, m, gsl_vector_complex_get(vectr, i));
            if (m==n) tr= gsl_complex_add(tr, gsl_vector_complex_get(vectr, i));
            i++;
        }
    }
    gsl_matrix_complex_scale(rho, gsl_complex_inverse(tr));
    gsl_matrix_complex_free(L);
    gsl_vector_complex_free(vectr);
}

double matrixnorm(gsl_matrix_complex *A){
    /*Compute the inf-norm of matrix A*/
    const size_t M = A->size1;
    double n = 0, t;
    int i, j;
    for(i=0; i<M; i++){
        for(j=0; j<M; j++){
            t = gsl_complex_abs(gsl_matrix_complex_get(A, i, j));
            if(t > n) n = t;
        }
    }
    return n;
}

double vectornorm(gsl_vector_complex *v){
    /*Compute the inf-norm of vector v*/
    const size_t M = v->size;
    double n = 0, t;
    int i;
    for(i=0; i<M; i++){
            t = gsl_complex_abs(gsl_vector_complex_get(v, i));
            if(t > n) n = t;
    }
    return n;
}

void powermethod (gsl_matrix_complex *A, gsl_vector_complex *v, unsigned int D){
    /*Compute the eigenvector whose eigenvalue is lower using the inverse power method*/
    int i, j, sig=1;
    double n;
    gsl_matrix_complex *L = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex *M = gsl_matrix_complex_alloc(D, D);
    gsl_vector_complex *w = gsl_vector_complex_alloc(D);
    double d = matrixnorm(A)*eps;
    for(i=0; i<D; i++){
        for(j=0; j<D; j++){
            if(i==j){
                gsl_matrix_complex_set(L, i, i, gsl_complex_add(gsl_matrix_complex_get(A, i, i), gsl_complex_rect(d, 0)));
            }else{
                gsl_matrix_complex_set(L, i, j, gsl_matrix_complex_get(A, i, j));
            }
        }
    }
    gsl_permutation *p = gsl_permutation_alloc(D);
    gsl_linalg_complex_LU_decomp(L, p, &sig);
    gsl_linalg_complex_LU_invert(L, p, M);
    gsl_matrix_complex_free(L);
    gsl_permutation_free(p);

    for(i=0; i<19; i++){
        gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, M, v, GSL_COMPLEX_ZERO, w);

        n = vectornorm(w);
        gsl_vector_complex_scale(w, gsl_complex_inverse(gsl_complex_rect(n, 0)));

        gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, A, w, GSL_COMPLEX_ZERO, v);
        n = vectornorm(v);
        for(j=0; j<D; j++){
            gsl_vector_complex_set(v, j, gsl_vector_complex_get(w, j));
        }
        if(n< eps)break;
    }
    gsl_matrix_complex_free(M);
    gsl_vector_complex_free(w);
}

double energydensity(gsl_matrix_complex *H, gsl_matrix_complex *R, gsl_matrix_complex *rho, double c, unsigned int D){
    gsl_matrix_complex *Q = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex *C = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex *aux = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex_memcpy(Q, H);
    gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, gsl_complex_rect(-0.5, 0), R, R, gsl_complex_rect(0, -1), Q);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, Q, R, GSL_COMPLEX_ZERO, C );
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, gsl_complex_rect(-1, 0), R, Q, GSL_COMPLEX_ONE, C );
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, R, R, GSL_COMPLEX_ZERO, aux);
    gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, GSL_COMPLEX_ONE, aux, aux, GSL_COMPLEX_ZERO, Q);
    gsl_matrix_complex_scale(Q, gsl_complex_rect(c, 0));
    gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, GSL_COMPLEX_ONE, C, C, GSL_COMPLEX_ZERO, aux);
    gsl_matrix_complex_add(Q, aux);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, rho, Q, GSL_COMPLEX_ZERO, aux);
    gsl_complex z = GSL_COMPLEX_ZERO;
    unsigned int i;
    for(i=0; i<D; i++){
        z = gsl_complex_add(z, gsl_matrix_complex_get(aux, i, i));
    }
    gsl_matrix_complex_free(Q);
    gsl_matrix_complex_free(C);
    gsl_matrix_complex_free(aux);
    return GSL_REAL(z);
}

double particlenumber(gsl_matrix_complex *R, gsl_matrix_complex *rho, unsigned int D){
    gsl_matrix_complex *aux = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex *C = gsl_matrix_complex_alloc(D, D);
    gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, GSL_COMPLEX_ONE, R, R, GSL_COMPLEX_ZERO, aux);
    gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, aux, rho, GSL_COMPLEX_ZERO, C);
    gsl_complex z = GSL_COMPLEX_ZERO;
    unsigned int i;
    for(i=0; i<D; i++){
        z = gsl_complex_add(z, gsl_matrix_complex_get(C, i, i));
    }
    gsl_matrix_complex_free(C);
    gsl_matrix_complex_free(aux);
    return GSL_REAL(z);
}

int gsl_linalg_complex_QR_decomp (gsl_matrix_complex * A){
    /*Compute the QR decomposition of a complex matrix
    Code by Jonny Taylor: http://lists.gnu.org/archive/html/help-gsl/2008-01/msg00079.html */
  const size_t M = A->size1;
  const size_t N = A->size2;
  gsl_vector_complex * tau = gsl_vector_complex_alloc(GSL_MIN (M, N));
      size_t i;

      for (i = 0; i < GSL_MIN (M, N); i++)
        {
          /* Compute the Householder transformation to reduce the j-th
             column of the matrix to a multiple of the j-th unit vector */

          gsl_vector_complex_view c_full = gsl_matrix_complex_column (A, i);
          gsl_vector_complex_view c = gsl_vector_complex_subvector (&(c_full.vector), i, M-i);

          gsl_complex tau_i = gsl_complex_conjugate(gsl_linalg_complex_householder_transform (&(c.vector)));

          gsl_vector_complex_set (tau, i, tau_i);

          /* Apply the transformation to the remaining columns and
             update the norms */

          if (i + 1 < N)
            {
              gsl_matrix_complex_view m = gsl_matrix_complex_submatrix (A, i, i + 1, M - i, N - (i + 1));
              gsl_linalg_complex_householder_hm (tau_i, &(c.vector), &(m.matrix));
            }
        }

      return GSL_SUCCESS;

}

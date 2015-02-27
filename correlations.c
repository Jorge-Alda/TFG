#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_rng.h>//Random number generator
#include <gsl/gsl_vector.h> //Vector handling
#include <gsl/gsl_matrix.h> //Matrix handling
#include <gsl/gsl_complex.h> //Complex numbers
#include <gsl/gsl_complex_math.h> //Complex number operations
#include <gsl/gsl_blas.h> //Linear algebra engine
#include <gsl/gsl_linalg.h>

double eps = 1e-6;

void reprM(gsl_matrix_complex *R0, unsigned int dim);
void reprv(gsl_vector_complex *v);
void extract(double *x, gsl_matrix_complex *H0, gsl_matrix_complex *R0, unsigned int D);
void expm(gsl_matrix_complex *A, gsl_complex alpha, int dimx);
void kronecker(gsl_matrix_complex *A, gsl_matrix_complex *B, unsigned int D, gsl_matrix_complex *prod);
void kronAI (gsl_matrix_complex *A, unsigned int D, gsl_matrix_complex *prod);
void kronIA (gsl_matrix_complex *A, unsigned int D, gsl_matrix_complex *prod);
void steadystatefinder(gsl_matrix_complex *H, gsl_matrix_complex *R, gsl_vector_complex *rho, unsigned int D);
double matrixnorm(gsl_matrix_complex *A);
double vectornorm(gsl_vector_complex *v);
void powermethod (gsl_matrix_complex *A, gsl_vector_complex *v, unsigned int D);
void spre(gsl_matrix_complex *A,  gsl_matrix_complex *preA, unsigned int D);
void spost(const gsl_matrix_complex *A, gsl_matrix_complex *postA, unsigned int trans, unsigned int D);

int main(){
    int D = 6, j, k;
    double x[2*D*D], dx=0.01;
    gsl_complex cff, cdd;

    FILE *f;
    f = fopen("KR_D6_c02.txt", "rt");
    for(j=0; j<2*(D)*(D); j++){
        fscanf(f, "%lf", &x[j]);
    }
    fclose(f);

    gsl_matrix_complex *H = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex *R = gsl_matrix_complex_alloc(D, D);
    gsl_matrix_complex *Q = gsl_matrix_complex_alloc(D, D);
    gsl_vector_complex *rho = gsl_vector_complex_alloc(D*D);
    gsl_vector_complex *vaux = gsl_vector_complex_alloc(D*D);
    gsl_vector_complex *vff = gsl_vector_complex_alloc(D*D);
    gsl_vector_complex *vdd = gsl_vector_complex_alloc(D*D);
    gsl_matrix_complex *T = gsl_matrix_complex_alloc(D*D, D*D);
    gsl_matrix_complex *RR = gsl_matrix_complex_alloc(D*D, D*D);
    gsl_matrix_complex *RI = gsl_matrix_complex_alloc(D*D, D*D);
    gsl_matrix_complex *IR = gsl_matrix_complex_alloc(D*D, D*D);
    gsl_matrix_complex *A = gsl_matrix_complex_alloc(D*D, D*D);
    gsl_matrix_complex *B = gsl_matrix_complex_alloc(D*D, D*D);

    extract(x, H, R, D);
    steadystatefinder(H, R, rho, D);

    gsl_matrix_complex_memcpy(Q, H);
    gsl_blas_zgemm(CblasConjTrans, CblasNoTrans, gsl_complex_rect(-0.5, 0), R, R, gsl_complex_rect(0, -1), Q);
    kronecker(R, R, D, RR);
    kronAI(Q, D, T);
    kronIA(Q, D, IR);
    gsl_matrix_complex_add(T, IR);
    gsl_matrix_complex_add(T, RR);
    kronAI(R, D, RI);
    kronIA(R, D, IR);

    gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, IR, rho, GSL_COMPLEX_ZERO, vff);
    gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, RR, rho, GSL_COMPLEX_ZERO, vdd);

    expm(T, gsl_complex_rect(dx, 0), D*D);

    f = fopen("corr_D6_c02.txt", "wt");
    gsl_matrix_complex_memcpy(A, T);
    for(j=0; j<5000; j++){
        gsl_blas_zgemm(CblasNoTrans, CblasNoTrans, GSL_COMPLEX_ONE, T, A, GSL_COMPLEX_ZERO, B);

        gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, B, vff, GSL_COMPLEX_ZERO, rho);
        gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, RI, rho, GSL_COMPLEX_ZERO, vaux);
        cff = GSL_COMPLEX_ZERO;
        for(k=0; k<D*D; k+=D+1){
            cff = gsl_complex_add(cff, gsl_vector_complex_get(vaux, k));
        }

        gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, B, vdd, GSL_COMPLEX_ZERO, rho);
        gsl_blas_zgemv(CblasNoTrans, GSL_COMPLEX_ONE, RR, rho, GSL_COMPLEX_ZERO, vaux);
        cdd = GSL_COMPLEX_ZERO;
        for(k=0; k<D*D; k+=D+1){
            cdd = gsl_complex_add(cdd, gsl_vector_complex_get(vaux, k));
        }

        fprintf(f, "%lf\t%lf\t%lf\n", j*dx, GSL_REAL(cff), GSL_REAL(cdd));

        gsl_matrix_complex_memcpy(A, B);
    }
    fclose(f);
    return 0;
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

void extract(double *x, gsl_matrix_complex *H0, gsl_matrix_complex *R0, unsigned int D){
    unsigned int a,n,m;
    for(m=0; m<D;m++){
        gsl_matrix_complex_set(H0, m, m, gsl_complex_rect(x[m],0));
    }
    a = D;
    for(m=0; m<D;m++){
        for(n=m+1; n<D;n++){
            gsl_matrix_complex_set(H0, m, n, gsl_complex_rect(x[a], x[a+1]));
            gsl_matrix_complex_set(H0, n, m, gsl_complex_rect(x[a], -x[a+1]));
            a += 2;
        }
    }

    a=0;
    for(m=0;m<D;m++){
        for(n=0;n<D;n++){
            gsl_matrix_complex_set(R0, n, m, gsl_complex_rect(x[a+D*D],0));
            a++;
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

void expm(gsl_matrix_complex *A, gsl_complex alpha, int dimx){
    /*Code by Sijo Joseph, http://stackoverflow.com/a/14718470 */
    int j,k=0;
    gsl_complex temp;
    gsl_matrix *matreal =gsl_matrix_alloc(2*dimx,2*dimx);
    gsl_matrix *expmatreal =gsl_matrix_alloc(2*dimx,2*dimx);
    //Converting the complex matrix into real one using A=[Areal, Aimag;-Aimag,Areal]
    for (j = 0; j < dimx;j++){
        for (k = 0; k < dimx;k++){
            temp=gsl_complex_mul(gsl_matrix_complex_get(A,j,k), alpha);
            gsl_matrix_set(matreal,j,k,GSL_REAL(temp));
            gsl_matrix_set(matreal,dimx+j,dimx+k,GSL_REAL(temp));
            gsl_matrix_set(matreal,j,dimx+k,GSL_IMAG(temp));
            gsl_matrix_set(matreal,dimx+j,k,-GSL_IMAG(temp));
        }
    }

    gsl_linalg_exponential_ss(matreal,expmatreal,.01);

    double realp;
    double imagp;
    for (j = 0; j < dimx;j++){
        for (k = 0; k < dimx;k++){
            realp=gsl_matrix_get(expmatreal,j,k);
            imagp=gsl_matrix_get(expmatreal,j,dimx+k);
            gsl_matrix_complex_set(A,j,k,gsl_complex_rect(realp,imagp));
        }
    }
    gsl_matrix_free(matreal);
    gsl_matrix_free(expmatreal);
}

void kronecker(gsl_matrix_complex *A, gsl_matrix_complex *B, unsigned int D, gsl_matrix_complex *prod){
    /*Compute Kronecker product A \otimes B* */
    int i, j, k, l;
    gsl_complex d;

    for (i = 0; i < D; i++){
        for (j = 0; j < D; j++){
            d = gsl_matrix_complex_get (A, i, j);
            for (k = 0; k < D; k++){
                for (l = 0; l < D; l++){
                    gsl_matrix_complex_set (prod, D*i+k, D*j+l, gsl_complex_mul(d, gsl_complex_conjugate(gsl_matrix_complex_get (B, k, l))) );
                }
            }
        }
    }
}

void kronAI (gsl_matrix_complex *A, unsigned int D, gsl_matrix_complex *prod){
    /*Compute Kronecker product A \otimes id */
    int i, j, k;
    gsl_complex d;

    for(i=0; i<D; i++){
        for(j=0; j<D; j++){
            d = gsl_matrix_complex_get(A, i, j);
            for(k=0; k<D; k++){
                gsl_matrix_complex_set(prod, D*i+k, D*j+k, d);
            }
        }
    }
}

void kronIA (gsl_matrix_complex *A, unsigned int D, gsl_matrix_complex *prod){
    /* Compute Kronecker product id \otimes A* */
    int i, j, k;
    gsl_complex d;

    for(i=0; i<D; i++){
        for(j=0; j<D; j++){
            d = gsl_complex_conjugate(gsl_matrix_complex_get(A, i, j));
            for(k=0; k<D; k++){
                gsl_matrix_complex_set(prod, i+k*D, j+k*D, d);
            }
        }
    }
}

void steadystatefinder(gsl_matrix_complex *H, gsl_matrix_complex *R, gsl_vector_complex *rho, unsigned int D){
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

    for(i=0; i<rho->size; i++){
        gsl_vector_complex_set(rho, i, gsl_complex_rect(gsl_rng_uniform(rng),gsl_rng_uniform(rng)));
    }
    powermethod(L, rho, D*D);
    gsl_matrix_complex_free(L);
    gsl_rng_free(rng);

    i=0;
    for(m=0; m<D*D; m+= (D+1)){
        tr= gsl_complex_add(tr, gsl_vector_complex_get(rho, m));
    }
    gsl_vector_complex_scale(rho, gsl_complex_inverse(tr));
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

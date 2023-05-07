
/*@Author : Nandan Joshi

* Implementation of a bicongugate Gradient asolver for linear equations
* Uses matrices in a sparse format
* Input matrix is in Compressed Space Row (CSR) format
* cuSparse allows matrix multiplications only in BSR (Block Spase Row) format
* CSR Converted to a BSR format within this code
* Details about CSR/BSR format : https://docs.nvidia.com/cuda/cusparse/index.html
*/


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include "helper_functions.h"  
#include "helper_cuda.h"       



/* Function solves a linear system using BCG
* Adopted from https://en.wikipedia.org/wiki/Biconjugate_gradient_stabilized_method
* Input CSR matrix needs to be converted to BSR first
* @param X: Holds the solution of BCG - passed by ref
* @param  RHS : Holds the RHS of the linear system
* @param rowPtr, colPtr, val : row offsets, col indices, and values of CSR storage
* @param mb,nb : Rows, Cols of the modified matrix in BSR
* @param nnzb : Number of non zero blocks in BSR
* @param vecSize : Size of the solution vector in BSR representation
* @param maxIt : Maximum no of iterations
*/

void BCGSolve(double* X, double* RHS, int* rowBSR, int* colBSR, double*valBSR, 
        cusparseHandle_t handle,  cusparseMatDescr_t descr_coeff, 
        cublasHandle_t handleBlas, 
        int mb, int nb, int nnzb, int vecSize, int maxit  ){

    /*Initialize variables*/


    double *R, *V, *T, *rw, *p;
    int dimBlock = 2;  //Block dimension hardcoded to 2

    double alpha = 1 ;
    double beta ;
    double alph;
    double bet;
    double omega = 1;
    double residual1 = 0;
    double residual2 = 0;
    double temp = 1;
    double rhop = 1; 
    double rho = 1;

    checkCudaErrors(cudaMalloc((double **)&R, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&p, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&rw, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&V, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&T, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMemset(V, 0, vecSize)); 


    const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW;

    /*Solve*/

   /*Step 1 : r = b - A.x*/ 
    alph = -1;
    bet = 0;

    /*-Calculate -Ax*/
    checkCudaErrors(cusparseDbsrmv(handle, dir_coeff, trans_coeff, mb, nb, nnzb, &alph,
                                    descr_coeff,valBSR, rowBSR, colBSR, dimBlock,X, 
                                    &bet, R));


    alph = 1;  
    /*Calculate b-Ax*/
    checkCudaErrors( cublasDaxpy( handleBlas, vecSize, 
                                    &alph,RHS, 1,R, 1));

    /*Assign r = b-Ax*/
    // checkCudaErrors (cudaMemcpy(Y, R,  (vecSize)*sizeof(double), 
    //                  cudaMemcpyDeviceToHost ));



    //Step 2: Set p=r and \tilde{r}=r
    checkCudaErrors(cublasDcopy(handleBlas,(vecSize), R, 1, p, 1));         //p=r
    checkCudaErrors(cublasDcopy(handleBlas, (vecSize), R, 1, rw,1));        //\tilde{r}=r
    checkCudaErrors(cublasDnrm2(handleBlas,(vecSize), R, 1, &residual1));   //Residual 



    /*/ Step 3: repeat until convergence */
    for (int i=0; i<maxit; i++){
        rhop = rho; 
        //Step 5.1 : Dot product (rw,r)
        checkCudaErrors(cublasDdot ( handleBlas, vecSize, rw, 1, R, 1, &rho));


        if (i > 0){
            //Step 5.2: \beta = (\rho_{i} / \rho_{i-1}) ( \alpha / \omega )
            beta= (rho/rhop)*(alpha/omega);

            //Step 5.3: p = r + \beta (p - \omega v)
            // -omega*v
            omega = -omega; 
            checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                            &omega,V, 1,p, 1));

            //Reset omega
            omega = -omega;


            // beta * (p - omega*v)
            checkCudaErrors(cublasDscal(handleBlas, vecSize, 
                                &beta,p, 1)); 

            //  r + beta*(p-omega*v)
            checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                        &alph,R, 1,p, 1)); 


            // checkCudaErrors (cudaMemcpy(Y, p,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

        }
            
            //Step 5.4 : v = A*p

            //A*p
            checkCudaErrors(cusparseDbsrmv(handle, dir_coeff, trans_coeff, mb, nb, nnzb, &alph,
                descr_coeff,valBSR, rowBSR, colBSR, dimBlock,p, 
                &bet, V));

            //Assign v = A*p
            // checkCudaErrors (cudaMemcpy(Y, V,  (vecSize)*sizeof(double), cudaMemcpyDeviceToHost ));

            /*Step 5.5 alpha = rho/ dot (\tilde{r}, p)*/

            //Store  dot (\tilde{r}, p) in alpha
            checkCudaErrors(cublasDdot ( handleBlas, vecSize, rw, 1, V, 1, &alpha));
            //Update alpha = rho/alpha
            alpha = rho/alpha; 


            /*Step 5.6/ s = r - \alpha * v */

            // Calculate -\alpha*v
            alpha = -alpha; 
            checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                        &alpha,V, 1,R, 1));  //S is stored in R to save memory


            //Reset alpha
            alpha = -alpha; 

            /*Step 5.5 X = X + p*alpha*/
            checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                &alpha,p, 1,X, 1));

            /*Step 5.7 : Check the residual of s*/
            checkCudaErrors(cublasDnrm2(handleBlas,(vecSize), R, 1, &residual2));

            if (residual2/residual1 < 1E-3){
                /*Converged*/
                break;
            }

        
        /*Step 5.9 T = A*s*/
        checkCudaErrors(cusparseDbsrmv(handle, dir_coeff, trans_coeff, mb, nb, nnzb, &alph,
            descr_coeff,valBSR, rowBSR, colBSR, dimBlock,R, 
            &bet, T));



        /*Step 5.10 omega = (T.T)/(T.R)*/
        checkCudaErrors(cublasDdot ( handleBlas, vecSize, T, 1, T, 1, &temp));  
        checkCudaErrors(cublasDdot ( handleBlas, vecSize, R, 1, T, 1, &omega));  

        omega = omega/temp; 

        //Step 5.11 *x = h + omega *s*/  

        checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                    &omega,R, 1,X, 1));


        /*Step 5.13 r = s - omega * t*/

        omega = -omega; 
        checkCudaErrors(cublasDaxpy(handleBlas, vecSize, 
                    &omega,T, 1,R, 1));
        //Reset omega
        omega = -omega; 


        /*Step 5.12 Check residual of R*/
        checkCudaErrors(cublasDnrm2(handleBlas,(vecSize), R, 1, &residual2));

         printf ("\n Residual %f \n", residual2/residual1);

        if (residual2/residual1 < 1E-3){
            /*Converged*/
            break;
        }
    }


}


/* Helper code to reserve buffer space for CSR to BSR conversion
* This function also calculates no of nonzero blocks in BSR
* @param rowPtr, colPtr, val : row offsets, col indices, and values of CSR storage
* @param rowBSR : row offsets for corresponding BSR format - passed by reference
* @param nnzb : non zero blocks in BSR - passed by reference
* @param  rows : Rows/Cols of the sparse matrix
* @param dimBlock : Dimension of block to be used in BSR
* @return pBuffer : Pointer to buffer space reserved for matrix operation
*/
void* getBSRDims(cusparseHandle_t handle, int* rowPtr, int* colPtr, double* val, 
        int* rowBSR, int*nnzb, 
        int rows,  int dimBlock
        ){


    cusparseMatDescr_t descr_coeff;
    cusparseMatDescr_t descr_coeff_2;
    
    int bufferSize = 0; 
    static void *pBuffer; 

    const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW;

    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ZERO));
    checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));

    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff_2));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff_2, CUSPARSE_INDEX_BASE_ZERO));
    checkCudaErrors(cusparseSetMatType(descr_coeff_2, CUSPARSE_MATRIX_TYPE_GENERAL));

    /* Obtain buffer size for CSR to BSR conversion*/
    checkCudaErrors(cusparseDcsr2gebsr_bufferSize(handle, dir_coeff, rows, rows,  
    descr_coeff, 
    val, rowPtr, colPtr, 
    dimBlock, dimBlock, 
    &bufferSize));

    /*Reserve buffer space*/
    checkCudaErrors(cudaMalloc((void**)&pBuffer, bufferSize));

    /*Get the number of nonzero blocks*/
    checkCudaErrors(cusparseXcsr2gebsrNnz(handle, dir_coeff, rows, rows, 
    descr_coeff, 
    rowPtr, colPtr,
    descr_coeff_2, 
    rowBSR, dimBlock, dimBlock, 
    nnzb, pBuffer));

    return pBuffer; 

}


/* Solves a linear system of euqations using BCG
* Accepts coefficient matrix in a sparse CSR storage
* Converts CSR into BSR before solving
* @param rowPtr, colPtr, val : row offsets, col indices, and values of coefficients
* @param Soln :Solution to the equations
* @param RHS : Right hand side of the equations
* @param nz : Nonzero elements in coeffieicent matrix 
* @param rows : rows,cols of coeff matrix
* @param maxIt : Maximum iterations*/

void LinearSolve( int* rowPtr, int* colPtr, double* val, 
                  double* Soln, double* RHS,                   
                  int rows, int nz, int maxit ){



    /*Variables to hold the info for BSR representation*/
    double*val_BSR = NULL;  //Values in BSR
    double* d_Y, *d_X;      //Soln and RHS resized for BSR dimensions
    int* row_BSR, *col_BSR; // Row offsets and col Indices in BSR
    int dimBlock = 2;       //Dim of BSR block - hardcoded
    int mb = (rows + dimBlock-1)/dimBlock;      //Resized no of rows in BSR
    int nb = (rows + dimBlock-1)/dimBlock;  //Resized no of cols in BSR
    int base; 
    int vecSize = rows + dimBlock-1; //Resized Soln dimension - BSR
    checkCudaErrors(cudaMalloc((double **)&d_X, (vecSize)*sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&d_Y, (vecSize)*sizeof(double)));


    /*Part 1 - Convert input from CSR to BSR format*/
    int bufferSize;
    void *pBuffer;
    int nnzb = 0; 
    cusparseHandle_t handle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&handle);
    
    cublasHandle_t handleBlas = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&handleBlas);

    const cusparseOperation_t trans_coeff  = CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseDirection_t dir_coeff = CUSPARSE_DIRECTION_ROW;

    cusparseMatDescr_t descr_coeff;
    cusparseMatDescr_t descr_coeff_2;
    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff, CUSPARSE_INDEX_BASE_ZERO));
    checkCudaErrors(cusparseSetMatType(descr_coeff, CUSPARSE_MATRIX_TYPE_GENERAL));

    checkCudaErrors(cusparseCreateMatDescr(&descr_coeff_2));
    checkCudaErrors(cusparseSetMatIndexBase(descr_coeff_2, CUSPARSE_INDEX_BASE_ZERO));
    checkCudaErrors(cusparseSetMatType(descr_coeff_2, CUSPARSE_MATRIX_TYPE_GENERAL));


    checkCudaErrors(cudaMalloc((void**)&row_BSR, sizeof(int) *(mb+1)));


    /*Reserve buffer space and obtain number of nonzero blocks in BSR*/
    pBuffer =  getBSRDims(handle, rowPtr, colPtr, val, row_BSR, &nnzb,
                    rows, dimBlock); 


    /*Allocate cols and vals based on nonzero blocks*/
    checkCudaErrors(cudaMalloc((void**)&col_BSR, sizeof(int)*(nnzb)));
    checkCudaErrors(cudaMalloc((void**)&val_BSR, sizeof(double)*(dimBlock*dimBlock)*(nnzb)));

    /*Convert CSR to BSR*/
    checkCudaErrors(cusparseDcsr2gebsr(handle, dir_coeff,rows, rows, descr_coeff, 
       val, rowPtr, colPtr, descr_coeff_2, val_BSR, row_BSR, col_BSR,        
        dimBlock, dimBlock, pBuffer));

    /*Transfer RHS and Soln to resized vectors*/
    checkCudaErrors(cublasDcopy(handleBlas,rows, RHS, 1, d_X, 1));
    checkCudaErrors(cublasDcopy(handleBlas, rows, Soln, 1, d_Y,1));

    /*Part 2 : Perform BCG solve*/
    BCGSolve(d_Y,  d_X,  row_BSR,  col_BSR, val_BSR, 
              handle, descr_coeff_2, handleBlas, 
               mb, nb, nnzb, vecSize, maxit); 

    /*Transfer result back to Soln*/
    checkCudaErrors(cublasDcopy(handleBlas, rows, d_Y, 1, Soln,1));

    checkCudaErrors(cudaFree(pBuffer));

}

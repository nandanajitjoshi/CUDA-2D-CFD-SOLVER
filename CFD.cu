#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>


/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

// Utilities and system includes
#include "helper_functions.h"  // helper for shared functions common to CUDA Samples
#include "helper_cuda.h"       // helper function CUDA error checking and initialization
/*Initialize CSR*/
#include "bcgstab.h"
#include "Mesh.h"
#include "Functions.h"
#include "PostPr.h"
//#include"Vertex.h"



/* Solve : Perorms a 2-D CFD simulation based on given inputs
* The function for mesh generation is called through ths function
* The output velocities are saved in files U_output and V_output
* Total number of recorded results has to be specified in arguments
* @param Lx,Ly : Dimension along x and y coordinates respectively
* @param Nx,Ny : No of grid points along X and Y direction respectively
* @param mu : Fluid viscosity
* @param rho : Fluid density
* @param delT : Size of time step
* @param nSteps : Number of timesteps
* @param nRecordedAfter : Timesteps after which each solution is recorded
*/

void Solve (double Lx, double Ly, int Nx,int Ny, 
            double mu, double rho, double delT, 
            int nSteps, int nRecordAfter){

    /*Initialize all variables*/
    int rows  = Ny*Nx; 
    int maxit = 50;  //Max BCG iterations - hardcoded
    int blockWidth = 128; // No of blocks per thread - hardcoded
    int nBlocks = rows/blockWidth + 1; 

    vertex* Domain = NULL;  //Mesh containing all geometry and inputs
    double* Uxlow, * Uxhigh, * Uxdiag;  //Tridiag coefficients for U along X
    double* Vxlow, *Vxhigh, *Vxdiag;   //Tridiag coefficients for V along X
    double* Uylow, * Uyhigh, * Uydiag; //Tridiag coefficients for U along Y
    double* Vylow, *Vyhigh, *Vydiag;  //Tridiag coefficients for V along Y
    double* U, *V, *P; //Final result vectors - U,V and P
    double* H_u_n_1, *H_v_n_1, *H_u_n, *H_v_n; //Convection terms in RHS
    double* D_u_n, *D_v_n;      //Diffusive terms in RHS
    double* RHS_u_n, *RHS_v_n;  //Final RHS for U and V for TDM equations
    int* P_rowPtr, *P_colPtr;   //Sparse representation of pressure coefficients 
    double* P_val, *P_RHS; 

    int nRecordedSteps = nSteps/nRecordAfter + 1; 

    /*Final result storage on the host*/
    double *U_Result= (double *)malloc(sizeof(double) * rows * nRecordedSteps );
    double *V_Result= (double *)malloc(sizeof(double) * rows * nRecordedSteps );


    cublasHandle_t handleBlas = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&handleBlas);
    checkCudaErrors (cublasStatus);

    /*Initialize all memory*/
    checkCudaErrors(cudaMalloc((vertex **)&Domain, rows *sizeof(vertex)));

    checkCudaErrors(cudaMalloc((double **)&Uxlow, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Uxhigh, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Uxdiag, rows *sizeof(double)));

    
    checkCudaErrors(cudaMalloc((double **)&Uylow, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Uyhigh, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Uydiag, rows *sizeof(double)));

    checkCudaErrors(cudaMalloc((double **)&Vxlow, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Vxhigh, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Vxdiag, rows *sizeof(double)));

    checkCudaErrors(cudaMalloc((double **)&Vylow, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Vyhigh, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&Vydiag, rows *sizeof(double)));

    checkCudaErrors(cudaMalloc((double **)&U, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&V, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&P, rows *sizeof(double)));

    checkCudaErrors(cudaMalloc((double **)&H_u_n_1, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&H_v_n_1, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&H_u_n, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&H_v_n, rows *sizeof(double)));
    
    checkCudaErrors(cudaMalloc((double **)&D_u_n, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&D_v_n, rows *sizeof(double)));


    checkCudaErrors(cudaMalloc((double **)&RHS_u_n, rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&RHS_v_n, rows *sizeof(double)));

    checkCudaErrors(cudaMalloc((double **)&P_rowPtr, (rows+1) *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&P_colPtr, 5*rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&P_val, 5*rows *sizeof(double)));
    checkCudaErrors(cudaMalloc((double **)&P_RHS, rows *sizeof(double)));


    /*Generate Mesh*/
    Mesh <<<nBlocks,blockWidth>>> (Lx,Ly,Nx,Ny, Domain);
    
    /*Obtain coeffiients for TDM equations for U and V*/
    getPCoeffs<<<nBlocks,blockWidth>>>(P_rowPtr, P_colPtr, P_val,
             Domain,  rows, Nx, Ny, rho, delT);

    genXCoeffs <<<nBlocks,blockWidth>>> (Uxlow,Uxhigh,Uxdiag,Domain, 
            rows, mu, rho, delT, 'U'); 

    genXCoeffs <<<nBlocks,blockWidth>>> (Vxlow,Vxhigh,Vxdiag,Domain,
             rows, mu, rho, delT, 'V'); 

    genYWithTrans (Uylow,Uyhigh,Uydiag,Domain, rows, mu, rho, delT,
         nBlocks, blockWidth, Ny, Nx, handleBlas, 'U' ); 

    genYWithTrans (Vylow,Vyhigh,Vydiag,Domain, rows, mu, rho, 
         delT, nBlocks, blockWidth, Ny, Nx, handleBlas, 'V' ); 

    
    /*Advance each timestep*/
    for (int t = 0; t<nSteps; t++){

        /* Step 1: Calculate convective and diffusive terms for TDM RHS*/
        calcH <<<nBlocks,blockWidth>>> (U,V,U, H_u_n, Domain, rows, Nx);
        calcH <<<nBlocks,blockWidth>>> (U,V,V, H_v_n, Domain, rows, Nx);
        calcD <<<nBlocks,blockWidth>>> (U,D_u_n, Domain, rows, Nx, 'U', delT, mu,rho );
        calcD <<<nBlocks,blockWidth>>> (V,D_v_n, Domain, rows, Nx, 'V', delT, mu, rho);

        /* Step 2 : Consolidate all RHS terms together*/
        updateRHS (handleBlas, H_u_n_1,  H_v_n_1,
        H_u_n,  H_v_n,  D_u_n, D_v_n, 
        RHS_u_n, RHS_v_n, delT, mu, rho, 
        rows); 


        /* Step 3 : Perform tridiaginal solve for U along X*/
        TriDiagSolve(Uxlow , Uxdiag, Uxhigh, RHS_u_n, rows);

        /*Step 4 : Peform tridiaginal solvefor U along Y*/

        // Stap 4.1 : Row-major to col-major
        XtoY<<<nBlocks, blockWidth>>> ( RHS_u_n, Ny, Nx, U);
        // Step 4.2 : Solve tridiag matrix
        TriDiagSolve(Uylow , Uydiag, Uyhigh, U, rows);
        //Step 4.3 :  Restore row-major from col-major
        YtoX<<<nBlocks, blockWidth>>> ( U, Ny, Nx, RHS_u_n);
        checkCudaErrors(cublasDcopy(handleBlas,(rows), RHS_u_n, 1, U, 1));

         /*Step 5 : Perform tridiaginal solve for V along X*/
        TriDiagSolve(Vxlow , Vxdiag, Vxhigh, RHS_v_n, rows);


        /* Step 6 : Peform tridiaginal solvefor V along Y*/
        // Step 6.1 : Row-major to col-major
        XtoY<<<nBlocks, blockWidth>>> ( RHS_v_n, Ny, Nx, V);
        // Step 6.2 : Solve tridiag matrix
        TriDiagSolve(Vylow , Vydiag, Vyhigh, V, rows);
        // Step 6.3 : Solve tridiag matrix
        YtoX<<<nBlocks, blockWidth>>> ( V, Ny, Nx, RHS_v_n);
        checkCudaErrors(cublasDcopy(handleBlas,(rows), RHS_v_n, 1, V, 1));

        /*Step 7 : Solve pressure poisson equation*/
        //Step 7.1 : Obtain RHS for pressure poisson
        update_PRHS <<<nBlocks, blockWidth>>> ( P_RHS, U, V, Domain, rows, Nx, Ny,
                 rho); 

        //Step 7.2 : Solve pressure poisson using BCG algorithm
        LinearSolve( P_rowPtr, P_colPtr, P_val, 
                    P, P_RHS,                   
                    rows, 5*rows, maxit );

        //Step 8 : Correct velocity field using updated pressure values
        velPressureCorrection <<<nBlocks, blockWidth>>>(P, U, V, Domain, 
            rows,  Nx, Ny, rho,  delT ); 

        
        //Step 9 : Prepare for next timestap
        //Step 9.1 : H_n_1 terms to be updated to current time
        checkCudaErrors(cublasDcopy(handleBlas,(rows), H_u_n, 1, H_u_n_1, 1));
        checkCudaErrors(cublasDcopy(handleBlas,(rows), H_v_n, 1, H_v_n_1, 1));
        //STep 9.2 : Reset the RHS
        checkCudaErrors(cudaMemset(RHS_u_n, 0, rows*sizeof(double)));
        checkCudaErrors(cudaMemset(RHS_v_n, 0, rows*sizeof(double)));

        if (t%nRecordAfter ==0){
        
            /*Store the result on host for postprocessing*/
            checkCudaErrors (cudaMemcpy( &U_Result[rows * (int)t/nRecordAfter], U,(rows)*sizeof(double), 
                cudaMemcpyDeviceToHost));

            checkCudaErrors (cudaMemcpy( &V_Result[rows * (int)t/nRecordAfter], V,(rows)*sizeof(double), 
                cudaMemcpyDeviceToHost));

            
        }

    }
    /*Write results in a file*/
    writeOutOutFile (U_Result, rows, nRecordedSteps, 'U'); 
    writeOutOutFile (V_Result, rows, nRecordedSteps, 'V'); 

}

    
 
    
/* Starting point of the program
* Takes all relevant inputs from command line
*/

int main(int argc, char** argv){
    float time; 
    int Ny = 50;
    int Nx = 50;
    double mu = 1E3; 
    double rho = 1;
    double delT = 0.00001;
    int nSteps = 201; 
    int nRecordAfter = 5; 

   clock_t start_t, end_t;
   double total_t;
    if (argc >= 2) {
            Ny = atoi(argv[1]);
            Nx = atoi(argv[2]);
            mu = atof(argv[3]); 
            rho = atof(argv[4]);
            delT = atof(argv[5]); 
            nSteps = atoi(argv[6]); 
            nRecordAfter = atoi(argv[7]); 
    }

    // validate command line arguments
    if (Ny< 3) {
        
        Ny = 3;
        printf("Warning: Problem size can't be less than 3\n");
        printf("The total number of threads will be modified  to 3\n");
    }

   start_t = clock(); 
   Solve(1,1,Nx,Ny, mu, rho, delT, nSteps, nRecordAfter);
   end_t = clock(); 
   total_t = (double)(end_t - start_t) / CLOCKS_PER_SEC;
    //time = callDiff( 1,Nx, Ny);
    printf ("The time taken for linear solve is \n");
    printf("%f", total_t);
}

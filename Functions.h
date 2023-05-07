//#include "Vertex.h"

__global__
void XtoY(double*, int, int , double*);

__global__
void YtoX(double*, int, int, double*); 

void genYWithTrans(double *, double *, double*, const vertex* , 
     int , double , double , double , int , int,
     int , int , cublasHandle_t, char); 


void TriDiagSolve(double*  , double*  , double* , double* , int ); 

__global__
void genXCoeffs(double *, double *, double*, const vertex* , 
     int , double , double , double , char ); 


__global__
void genYCoeffs(double *, double *, double*, const vertex*, 
     int , double , double , double , char );

__global__
void calcH(const double *, const double *, const double*, double* , 
    const vertex* , int , int ); 

__global__
void calcD(const double *, double*,  const vertex* , int , int , 
    char , double , double , double );


void updateRHS(cublasHandle_t , double* , double* ,
     double* , double* , double* , double* , 
     double* , double* , double , double , double , 
     int  ); 

__global__
void getPCoeffs(int* , int* , double* ,
                 const vertex* , int , int , int ,
                 double , double ); 

__global__
void update_PRHS(double* , double* , double* ,
                 const vertex* , int , int , int ,
                 double ); 

__global__
void velPressureCorrection (double*, double* , double* ,const vertex* , 
        int , int , int ,double , double  ); 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include<fstream>



void writeOutOutFile (double* resArray, int rows, int nRecordedSteps, char VarId){
    
    FILE *fp1; 
    char name[] = " Output.csv"; 
    char fullName[25]; 
    fullName [0] = VarId; 
    fullName[1] = '\0'; 
    strcat (fullName, name);    

    printf ("%d\n", nRecordedSteps) ; 

    fp1 = fopen (fullName, "w"); 

    if (fp1 == NULL)
    {
        printf("Error while opening the file.\n");
    }

    for (int i = 0; i< nRecordedSteps; i++){
        fprintf (fp1, "%d,", i); 
        for (int j = 0; j<rows; j++){
            fprintf (fp1, "%0.2f", resArray[j + i*rows]); 
            fprintf (fp1, ","); 

        }
        fprintf (fp1, "\n");
    }

    fclose (fp1); 
}
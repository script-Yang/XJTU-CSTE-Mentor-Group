#include<iostream>
using namespace std;
int main()
{
    int n, k, i;
    cin>>n>>k;
    
    
    int array[n];
    for(i = 0; i < n; i++)  
        array[i] = 1;
    
    
    
    for(i = 2; i <= k; i++)
    {
        for(int p = 1; p < n; p++)
        {
            if((p+1) % i == 0)      
                array[p] = !array[p];       
        }
    }
    for(i = 0; i < n; i++)
    {
        if(array[i])       
            cout<<i+1<<" ";
    }
    return 0;
    
    
}

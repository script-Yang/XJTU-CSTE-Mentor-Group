#include<bits/stdc++.h>
 
int a[1007];
 
int main(){
    int n, k;
    std::cin >> n >> k;
    for(int i = 1; i <= k; ++i)
        for(int j = i; j <= n; j+=i)
            a[j]^=1;
    for(int i = 1; i <= n; ++i)
        if(a[i])std::cout<< i << ' ';
    return 0;
}

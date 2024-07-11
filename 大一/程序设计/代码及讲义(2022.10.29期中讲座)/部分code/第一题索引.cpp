#include<bits/stdc++.h>
using namespace std;
int main(){
	int n;
	cin>>n;
	int* a = new int[n];
	for(int i = 0;i<n;i++)
		cin>>a[i];
	int s;
	cin>>s;
	int flag = 0;
	for(int i = 0;i<n;i++){
		for(int j = i+1;j<n;j++){
			if((a[i]+a[j] == s)){
				cout << '[' << i << ',' << j << ']';
				flag = 1;
				break;
			}
		} 
		if(flag)break;
	}
	if(!flag) cout<<"Wrong input!"<<endl; 
	delete []a;
	return 0;
}

#include<bits/stdc++.h>
using namespace std;
#define max_len 100

string save_str[max_len];

int cnt_str[max_len];

int pos;
int main(){
	string s;
	while(cin>>s){
		int flag = 0;
		for(int i = 0;i<pos;i++){
			if(s==save_str[i]){
				cnt_str[i]++;
				flag = 1;
				break;
			}
		}
        
		if(!flag){
			save_str[pos] = s;
			cnt_str[pos]++;
			pos++;
		}
        
	}
    
    
    
    
    
    
    
    
	int p = 0;
	for(int i = 1; i < pos;i++){
		bool f1 = (cnt_str[i] >  cnt_str[p]);
		bool f2 = (cnt_str[i] == cnt_str[p]) && (save_str[i]<save_str[p]);
        
		if(f1||f2) p = i;
	}
    
    
    
    
    
    
	cout << save_str[p] << ' ' << cnt_str[p] << '\n';
	return 0;
}

#include<iostream>
#include<string>
#include<vector>
using namespace std;
int main()
{
    string s;
    cin>>s;
    vector<string> res;
    string temp;
    for(int i=0;i<s.size();i++)
    {

        if(s[i]<='9'&&s[i]>='0')
        {
            temp+=s[i]
        }
        else
        {
            if(temp.size()!=0)
                res.push_back(temp);
                temp="";
        }
    }
    for(auto s: res)
        cout<<s<<endl;
    return 0;
}
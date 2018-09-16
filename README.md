### Longest Palindromic Substring

```
public String longestPalindrome(String a) {
        if(a==null) return a;
        String largest = a.substring(0,1);
        for(int i=0;i<a.length()-1;i++){
            String pallindrome = longestPalindrome(a,i,i);
            if(pallindrome!=null && pallindrome.length()>largest.length()){
                largest = pallindrome;
            }
            pallindrome = longestPalindrome(a,i,i+1);
            if(pallindrome!=null && pallindrome.length()>largest.length()){
                largest = pallindrome;
            }
        }
        return largest;
    }
    
    String longestPalindrome(String s, int left, int right){
        if(left>right) return null;
        while( left>=0 && right<s.length() && s.charAt(left) == s.charAt(right)){
            left--;
            right++;
        }
        return s.substring(left+1,right);
}
```

### Max Sum Contiguous Subarray ###

```
public int maxSubArray(final List<Integer> a) {
         int sum = Integer.MIN_VALUE;
        int last = 0;
        
        for (int num : a) {
            
            last += num;
            sum = Math.max(sum, last);
            if (last < 0)
                last = 0;
        }
        
        return sum;
}
```

### String to Integer (atoi)

```
public int atoi(String str) {
	if (str == null || str.length() < 1)
		return 0;
 
	// trim white spaces
	str = str.trim();
 
	char flag = '+';
 
	// check negative or positive
	int i = 0;
	if (str.charAt(0) == '-') {
		flag = '-';
		i++;
	} else if (str.charAt(0) == '+') {
		i++;
	}
	// use double to store result
	double result = 0;
 
	// calculate value
	while (str.length() > i && str.charAt(i) >= '0' && str.charAt(i) <= '9') {
		result = result * 10 + (str.charAt(i) - '0');
		i++;
	}
 
	if (flag == '-')
		result = -result;
 
	// handle max and min
	if (result > Integer.MAX_VALUE)
		return Integer.MAX_VALUE;
 
	if (result < Integer.MIN_VALUE)
		return Integer.MIN_VALUE;
 
	return (int) result;
}
```

### Longest Substring Without Repeating Characters

```
public class Solution {
    public int lengthOfLongestSubstring(String s) {
        int n = s.length(), ans = 0;
        Map<Character, Integer> map = new HashMap<>(); // current index of character
        // try to extend the range [i, j]
        for (int j = 0, i = 0; j < n; j++) {
            if (map.containsKey(s.charAt(j))) {
                i = Math.max(map.get(s.charAt(j)), i);
            }
            ans = Math.max(ans, j - i + 1);
            map.put(s.charAt(j), j + 1);
        }
        return ans;
    }
}
```

### Container With Most Water

```
public class Solution {
    public int maxArea(int[] height) {
        int maxarea = 0, l = 0, r = height.length - 1;
        while (l < r) {
            maxarea = Math.max(maxarea, Math.min(height[l], height[r]) * (r - l));
            if (height[l] < height[r])
                l++;
            else
                r--;
        }
        return maxarea;
    }
}
```

### Longest Common Prefix

```
 public String longestCommonPrefix(String[] strs) {
    if (strs.length == 0) return "";
    String prefix = strs[0];
    for (int i = 1; i < strs.length; i++)
        while (strs[i].indexOf(prefix) != 0) {
            prefix = prefix.substring(0, prefix.length() - 1);
            if (prefix.isEmpty()) return "";
        }        
    return prefix;
}

Time complexity : O(S) , where S is the sum of all characters in all strings. 
```

```
public String longestCommonPrefix(String[] strs) {
    if (strs == null || strs.length == 0)
        return "";
    int minLen = Integer.MAX_VALUE;
    for (String str : strs)
        minLen = Math.min(minLen, str.length());
    int low = 1;
    int high = minLen;
    while (low <= high) {
        int middle = (low + high) / 2;
        if (isCommonPrefix(strs, middle))
            low = middle + 1;
        else
            high = middle - 1;
    }
    return strs[0].substring(0, (low + high) / 2);
}

private boolean isCommonPrefix(String[] strs, int len){
    String str1 = strs[0].substring(0,len);
    for (int i = 1; i < strs.length; i++)
        if (!strs[i].startsWith(str1))
            return false;
    return true;
}
Time complexity : O(S⋅log(n)), where S is the sum of all characters in all strings.
```

### 3Sum

```
public List<List<Integer>> threeSum(int[] nums) {
        Arrays.sort(nums);
        HashMap<Integer, Integer> sl = new HashMap<>();
        List<List<Integer>> res = new ArrayList<>();
        for(int i=0;i<nums.length;i++)
            sl.put(nums[i], i);      // overwriting i to hold the latest position of repeating number 
        for(int i=0;i<nums.length-2;i++)
        {
            for(int j=i+1;j<nums.length-1;j++)
            {
               int target = 0-nums[i]-nums[j];
                if(sl.containsKey(target) && sl.get(target)>j)
                {
                    j=sl.get(nums[j]);
                    res.add(Arrays.asList(nums[i], nums[j], target));
                }
            }
            i=sl.get(nums[i]);  // To remove duplicates
        }
        return res;
    }
```

### Valid Parentheses

```
if(s == null || s.length() < 1) {
         return true;
     }   
     
    Stack<Character> stack = new Stack<>();
    
    for(char c : s.toCharArray()) {
        switch(c) {
            case ')':
                if(stack.isEmpty() || stack.pop() != '(') return false;
                break;
            case ']':
                if(stack.isEmpty() || stack.pop() != '[') return false;
                break;
            case '}':
                if(stack.isEmpty() || stack.pop() != '{') return false;``
                break;
                
            default:
                stack.push(c);
                break;
        }
    }
    
  return stack.isEmpty();
```

### Count and say

```
public String countAndSay(int n) {
        String s = "1";
        StringBuilder sb = new StringBuilder();
        for(int i = 2; i <= n; i++) {
            for(int j = 0; j < s.length(); j++) {
                char tmp = s.charAt(j);
                int cnt = 1;
                while(j < s.length() - 1 && s.charAt(j+1) == s.charAt(j)) {
                    cnt++;
                    j++;
                }
                sb.append(cnt).append(tmp);
            }
            s = sb.toString();
            sb = new StringBuilder();
        }
        return s;
    }
```

### Binary search Tree(Insert and search)

```
// Function to create a new Node in heap
BstNode* GetNewNode(int data) {
	BstNode* newNode = new BstNode();
	newNode->data = data;
	newNode->left = newNode->right = NULL;
	return newNode;
}

// To insert data in BST, returns address of root node 
BstNode* Insert(BstNode* root,int data) {
	if(root == NULL) { // empty tree
		root = GetNewNode(data);
	}
	// if data to be inserted is lesser, insert in left subtree. 
	else if(data <= root->data) {
		root->left = Insert(root->left,data);
	}
	// else, insert in right subtree. 
	else {
		root->right = Insert(root->right,data);
	}
	return root;
}
//To search an element in BST, returns true if element is found
bool Search(BstNode* root,int data) {
	if(root == NULL) {
		return false;
	}
	else if(root->data == data) {
		return true;
	}
	else if(data <= root->data) {
		return Search(root->left,data);
	}
	else {
		return Search(root->right,data);
	}
}
int main() {
	BstNode* root = NULL;  // Creating an empty tree
	/*Code to test the logic*/
	root = Insert(root,15);	
	root = Insert(root,10);	
	root = Insert(root,20);
	root = Insert(root,25);
	root = Insert(root,8);
	root = Insert(root,12);
	// Ask user to enter a number.  
	int number;
	cout<<"Enter number be searched\n";
	cin>>number;
	//If number is found, print "FOUND"
	if(Search(root,number) == true) cout<<"Found\n";
	else cout<<"Not Found\n";
}
```

### Binary tree BFS and DFS

```
void LevelOrder(Node *root) {
	if(root == NULL) return;
	queue<Node*> Q;
	Q.push(root);  
	//while there is at least one discovered node
	while(!Q.empty()) {
		Node* current = Q.front();
		Q.pop(); // removing the element at front
		cout<<current->data<<" ";
		if(current->left != NULL) Q.push(current->left);
		if(current->right != NULL) Q.push(current->right);
	}
}

//Function to visit nodes in Preorder
void Preorder(struct Node *root) {
	// base condition for recursion
	// if tree/sub-tree is empty, return and exit
	if(root == NULL) return;

	printf("%c ",root->data); // Print data
	Preorder(root->left);     // Visit left subtree
	Preorder(root->right);    // Visit right subtree
}

//Function to visit nodes in Inorder
void Inorder(Node *root) {
	if(root == NULL) return;

	Inorder(root->left);       //Visit left subtree
	printf("%c ",root->data);  //Print data
	Inorder(root->right);      // Visit right subtree
}

//Function to visit nodes in Postorder
void Postorder(Node *root) {
	if(root == NULL) return;

	Postorder(root->left);    // Visit left subtree
	Postorder(root->right);   // Visit right subtree
	printf("%c ",root->data); // Print data
}


```

### Decode message

<img width="964" src="https://user-images.githubusercontent.com/5608772/45596270-35694a00-b9d7-11e8-820d-08ee823dfa31.png">

http://www.youtube.com/watch?v=qli-JCrSwuk

### Binary search

```
   int binarySearch(int arr[], int x) 
    { 
        int l = 0, r = arr.length - 1; 
        while (l <= r) 
        { 
            int m = l + (r-l)/2; 
  
            // Check if x is present at mid 
            if (arr[m] == x) 
                return m; 
  
            // If x greater, ignore left half 
            if (arr[m] < x) 
                l = m + 1; 
  
            // If x is smaller, ignore right half 
            else
                r = m - 1; 
        } 
  
        // if we reach here, then element was  
        // not present 
        return -1; 
    }
```

### First and last occurence of a number in an array

```
FirstOcc(Array a, size n, search_value s){
	low=0;
	high=n-1;
        result=-1
	while(low<=high){
		mid=(low+high)/2;
		if(a[mid]<s)
		  	 low=mid+1;
		else if(a[mid]>s) 
			high=mid-1; 
		else if(a[mid]==s) {
		 	result=mid; high=mid-1;}
	}
	return result; 
}

lastOcc(Array a, size n, search_value s){
	low=0;
	high=n-1;
        result=-1
	while(low<=high){
		mid=(low+high)/2;
		if(a[mid]<s)
		   low=mid+1;
		else if(a[mid]>s)
		    high=mid-1;
		else if(a[mid]==s)
		 { result=mid; low=mid+1;}
	}
     return ressult;
}
```
### How many times array is rotated

<img width="575" alt="screen shot 2018-09-16 at 5 58 35 pm" src="https://user-images.githubusercontent.com/5608772/45596489-3354ba80-b9da-11e8-88e5-c8f8f66dc3fe.png">

### Search in circularly sorted array

<img width="576" alt="screen shot 2018-09-16 at 6 15 45 pm" src="https://user-images.githubusercontent.com/5608772/45596644-97787e00-b9dc-11e8-9c50-f0cc5e8f9522.png">

### Knapsack solution

```
W = [2,3,4,5]
V= [1,3,2,9]
int[n][c] memo= null;
void KS(n,c){
  int result =0;
  if(memo[n][v]!=null) return memo[n][v];
  if(n==0 || c==0) result= 0;
  else if(w[n]>c) result = KS(n-1,c); 
  else{
    temp1 = KS(n-1,c);
    temp2 = v[n]+ KS(n-1,w[n]-c);
    result = Max(temp1,temp2);
  }
  memo[n][c] = result;
  return result;
}
```




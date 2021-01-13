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
### All Palindromes of a string

```
// expand in both directions of low and high to find all palindromes
	public static void expand(String str, int low, int high, Set<String> set)
	{
		// run till str[low.high] is a palindrome
		while (low >= 0 && high < str.length()
				&& str.charAt(low) == str.charAt(high))
		{
			// push all palindromes into the set
			set.add(str.substring(low, high + 1));

			// expand in both directions
			low--;
			high++;
		}
	}

	// Function to find all unique palindromic substrings of given string
	public static void allPalindromicSubStrings(String str)
	{
		// create an empty set to store all unique palindromic substrings
		Set<String> set = new HashSet<>();

		for (int i = 0; i < str.length(); i++)
		{
			// find all odd length palindrome with str[i] as mid point
			expand(str, i, i, set);

			// find all even length palindrome with str[i] and str[i+1]
			// as its mid points
			expand(str, i, i + 1, set);
		}

		// print all unique palindromic substrings
		System.out.print(set);
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

### Max product subarray

```
public class Solution {
    public int maxProduct(int[] A) {
        if (A == null || A.length == 0) {
            return 0;
        }
        int max = A[0], min = A[0], result = A[0];
        for (int i = 1; i < A.length; i++) {
            int temp = max;
            max = Math.max(Math.max(max * A[i], min * A[i]), A[i]);
            min = Math.min(Math.min(temp * A[i], min * A[i]), A[i]);
            if (max > result) {
                result = max;
            }
        }
        return result;
    }
}
```

### Subarray with K Sum

```
public int maxSubArrayLen(int[] nums, int k) {
    HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
 
    int max = 0;
 
    int sum=0;
    for(int i=0; i<nums.length; i++){
        sum += nums[i];
 
        if(sum==k){
            max = Math.max(max, i+1);
        }  
 
        int diff = sum-k;
 
        if(map.containsKey(diff)){
            max = Math.max(max, i-map.get(diff));
        }
 
        if(!map.containsKey(sum)){
            map.put(sum, i);
        }
    }
 
 
    return max;
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
    if(s==null||s.length()==0){
        return 0;
    }
    int result = 0;
    int k=0;
    HashSet<Character> set = new HashSet<Character>();
    for(int i=0; i<s.length(); i++){
        char c = s.charAt(i);
        if(!set.contains(c)){
            set.add(c);
            result = Math.max(result, set.size());
        }else{
            while(k<i){
                if(s.charAt(k)==c){
                    k++;
                    break;
                }else{
                    set.remove(s.charAt(k));
                    k++;
                }
            }
        }  
    }
 
    return result;
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
/ Returns the maximum value that can be put in a knapsack of capacity W 
    static int knapSack(int W, int wt[], int val[], int n) 
    { 
         int i, w; 
     int K[][] = new int[n+1][W+1]; 
       
     // Build table K[][] in bottom up manner 
     for (i = 0; i <= n; i++) 
     { 
         for (w = 0; w <= W; w++) 
         { 
             if (i==0 || w==0) 
                  K[i][w] = 0; 
             else if (wt[i-1] <= w) 
                   K[i][w] = max(val[i-1] + K[i-1][w-wt[i-1]],  K[i-1][w]); 
             else
                   K[i][w] = K[i-1][w]; 
         } 
      } 
       
      return K[n][W]; 
    } 
    
    
    //Recursion and dep
    int rec(int idx, int weight) {
	if(weight >= W){
		return 0;
	}
	if(idx > N){
		return 0;
	}
	if(dp[idx][weight]!=-1)return dp[idx][weight]
	int ans = 0;
	
	ans = max(ans, rec(idx+1, weight));
	if(weight+w[idx] <= W){
		ans = max(ans, value[idx] + rec(idx+1, weight+w[idx]));	
	}
	
	return dp[idx][weight] = ans;
}
```

## Reverse a string

```
String arr = "abc";
char temp;
char[] arr = inpStr.toCharArray();
int len = arr.length;
for(int i=0;i<len/2;i++,len--){
  temp = arr[i];
  arr[i]=arr[len-1];
  arr[len-1] = temp;
}
```

```
public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }
```

## Check if a string is pallindrome

```
boolean isPalindrome(String s) {
  int n = s.length();
  for (int i = 0; i < (n/2); ++i) {
     if (s.charAt(i) != s.charAt(n - i - 1)) {
         return false;
     }
  }

  return true;
}
```

## Insert a node at nth position

```
void Insert(int data, int n)
{
    int i;
    struct Node* temp1  = (struct Node*) malloc(sizeof(struct Node));
    temp1->data = data;
    temp1->next = NULL;
    if(n == 1)
    {
        temp1->next = head;
        head = temp1;
        return;
    }
    struct Node* temp2 = head;
    for(i=0;i<n-2;i++)
    {
        temp2 = temp2->next;
    }
 
    temp1->next = temp2->next;
    temp2->next = temp1;
}
```

## Reverse a linked list - Iterative method


<img width="447" alt="screen shot 2018-09-16 at 11 04 18 pm" src="https://user-images.githubusercontent.com/5608772/45599195-ecc88580-ba04-11e8-8e01-c2b7b58c3279.png">

## Combination sum 

https://youtu.be/zKwwjAkaXLI?t=804

## Coin change DP

```
public static int dynamic(int[] v, int amount) {
		int[][] solution = new int[v.length + 1][amount + 1];

		// if amount=0 then just return empty set to make the change
		for (int i = 0; i <= v.length; i++) {
			solution[i][0] = 1;
		}

		// if no coins given, 0 ways to change the amount
		for (int i = 1; i <= amount; i++) {
			solution[0][i] = 0;
		}

		// now fill rest of the matrix.

		for (int i = 1; i <= v.length; i++) {
			for (int j = 1; j <= amount; j++) {
				// check if the coin value is less than the amount needed
				if (v[i - 1] <= j) {
					// reduce the amount by coin value and
					// use the subproblem solution (amount-v[i]) and
					// add the solution from the top to it
					solution[i][j] = solution[i - 1][j]
							+ solution[i][j - v[i - 1]];
				} else {
					// just copy the value from the top
					solution[i][j] = solution[i - 1][j];
				}
			}
		}
		return solution[v.length][amount];
	}

	public static void main(String[] args) {
		int amount = 5;
		int[] v = { 1, 2, 3 };
		System.out.println("By Dynamic Programming " + dynamic(v, amount));
	}
	
	//Recursion
	int rec(int idx, int sum) {
	if(sum > SUM){
		return 0;
	}
	if(sum == SUM){
		return 1;
	}
	if(idx > N){
		return 0;
	}
	
	if(dp[idx][sum]!=-1)return dp[idx][sum];
	
	int ans = 0;
	
	ans = rec(idx+1,sum);
	ans = rec(idx, sum + S[idx]) + ans;
	
	return dp[idx][sum] = ans;
	
}

}
```

## Infix/Prefix/Postfix

Postfix 
```
// Function to evaluate Postfix expression and return output
int EvaluatePostfix(string expression)
{
	// Declaring a Stack from Standard template library in C++. 
	stack<int> S;

	for(int i = 0;i< expression.length();i++) {

		// Scanning each character from left. 
		// If character is a delimitter, move on. 
		if(expression[i] == ' ' || expression[i] == ',') continue; 

		// If character is operator, pop two elements from stack, perform operation and push the result back. 
		else if(IsOperator(expression[i])) {
			// Pop two operands. 
			int operand2 = S.top(); S.pop();
			int operand1 = S.top(); S.pop();
			// Perform operation
			int result = PerformOperation(expression[i], operand1, operand2);
			//Push back result of operation on stack. 
			S.push(result);
		}
		else if(IsNumericDigit(expression[i])){
			// Extract the numeric operand from the string
			// Keep incrementing i as long as you are getting a numeric digit. 
			int operand = 0; 
			while(i<expression.length() && IsNumericDigit(expression[i])) {
				// For a number with more than one digits, as we are scanning from left to right. 
				// Everytime , we get a digit towards right, we can multiply current total in operand by 10 
				// and add the new digit. 
				operand = (operand*10) + (expression[i] - '0'); 
				i++;
			}
			// Finally, you will come out of while loop with i set to a non-numeric character or end of string
			// decrement i because it will be incremented in increment section of loop once again. 
			// We do not want to skip the non-numeric character by incrementing i twice. 
			i--;

			// Push operand on stack. 
			S.push(operand);
		}
	}
	// If expression is in correct format, Stack will finally have one element. This will be the output. 
	return S.top();
}
```

## Graph Coloring

<img width="949" alt="screen shot 2018-09-19 at 9 47 29 pm" src="https://user-images.githubusercontent.com/5608772/45766662-bdf51e00-bc55-11e8-9b94-3e959c5684f9.png">

## Fence painting

```
int paintFence(int n, int k){
if(n==0) return 0;
if(n==1) return k;
int[] dp = new int[n];
dp[1] = k;
int same = 0;
int diff = k;
for(int i=2;i<=n;i++){
  same = diff;
  diff = dp[i-1]*(k-1) // 
  dp[i] = same +diff;
}
return dp[n];
}
```

### Rotate an array right

```
public class Solution {
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }
    public void reverse(int[] nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }
}
```

## Rotate an array left

```
rotate(arr[], d, n)
  reverse(arr[], 1, d) ;
  reverse(arr[], d + 1, n);
  reverse(arr[], 1, n);
```

## Local Minima

```
 public static int localMinUtil(int[] arr, int low, 
                                   int high, int n)
    {
         
        // Find index of middle element
        int mid = low + (high - low) / 2;
         
         // Compare middle element with its neighbours
        // (if neighbours exist)
        if(mid == 0 || arr[mid - 1] > arr[mid] && mid == n - 1 || 
           arr[mid] < arr[mid + 1])
                return mid;
         
        // If middle element is not minima and its left
        // neighbour is smaller than it, then left half
        // must have a local minima.
        else if(mid > 0 && arr[mid - 1] < arr[mid])
                return localMinUtil(arr, low, mid - 1, n);
         
        // If middle element is not minima and its right
        // neighbour is smaller than it, then right half
        // must have a local minima.
        return localMinUtil(arr, mid + 1, high, n);
    }
```

## Minimum number of platforms

```
// Program to find minimum number of platforms 
 
import java.util.*;
 
class GFG {
  
// Returns minimum number of platforms reqquired
static int findPlatform(int arr[], int dep[], int n)
{
   // Sort arrival and departure arrays
   Arrays.sort(arr);
   Arrays.sort(dep);
  
   // plat_needed indicates number of platforms
   // needed at a time
   int plat_needed = 1, result = 1;
   int i = 1, j = 0;
  
   // Similar to merge in merge sort to process 
   // all events in sorted order
   while (i < n && j < n)
   {
      // If next event in sorted order is arrival, 
      // increment count of platforms needed
      if (arr[i] <= dep[j])
      {
          plat_needed++;
          i++;
  
          // Update result if needed 
          if (plat_needed > result) 
              result = plat_needed;
      }
  
      // Else decrement count of platforms needed
      else
      {
          plat_needed--;
          j++;
      }
   }
  
   return result;
}
```

### Undo/Redo
https://codereview.stackexchange.com/questions/172662/command-pattern-with-undo-returning-response-in-invoker-and-command-class-or-c

### Find K smallest/largest element in an array

https://www.geeksforgeeks.org/?p=2392/

```
public List<String> topKFrequent(String[] words, int k) {
        List<String> ans = new ArrayList<>();
        if(words == null || words.length == 0) {
            return ans;
        }
        Map<String, Integer> map = new HashMap<>();
        for(String word : words) {
            map.put(word,map.getOrDefault(word, 0) + 1);
        }
        PriorityQueue<String> pq = new PriorityQueue<>
            ((a,b) -> map.get(b) ==  map.get(a)? b.compareTo(a) : map.get(a) - map.get(b));
        for (String word: map.keySet()) {
            pq.offer(word);
            if (pq.size() > k) pq.poll();
        }
        while (!pq.isEmpty()) ans.add(0,pq.poll());
        return ans;
    }
    
    O(nlogk)
```

### Lowest common ansestor

```
Node lca(Node node, int n1, int n2)  
    { 
        if (node == null) 
            return null; 
   
        // If both n1 and n2 are smaller than root, then LCA lies in left 
        if (node.data > n1 && node.data > n2) 
            return lca(node.left, n1, n2); 
   
        // If both n1 and n2 are greater than root, then LCA lies in right 
        if (node.data < n1 && node.data < n2)  
            return lca(node.right, n1, n2); 
   
        return node; 
    } 
```

### Find Lowest common ansestor of Binary tree not binary search

```
Node findLCA(Node node, int n1, int n2) 
    { 
        // Base case 
        if (node == null) 
            return null; 
  
        // If either n1 or n2 matches with root's key, report 
        // the presence by returning root (Note that if a key is 
        // ancestor of other, then the ancestor key becomes LCA 
        if (node.data == n1 || node.data == n2) 
            return node; 
  
        // Look for keys in left and right subtrees 
        Node left_lca = findLCA(node.left, n1, n2); 
        Node right_lca = findLCA(node.right, n1, n2); 
  
        // If both of the above calls return Non-NULL, then one key 
        // is present in once subtree and other is present in other, 
        // So this node is the LCA 
        if (left_lca!=null && right_lca!=null) 
            return node; 
  
        // Otherwise check if left subtree or right subtree is LCA 
        return (left_lca != null) ? left_lca : right_lca; 
    }
```

### Distance between two nodes of a binary tree

```
Node lca = findLCA(root,n1,n2);
int l1 = getLevel(lca,n1,0);
int l2 = getLevel(lca,n2,0);
return l1+l2;
```

### Height of a binary tree

```
int height(Node root) 
    { 
        if (root == null) 
           return 0; 
        else
        { 
            /* compute  height of each subtree */
            int lheight = height(root.left); 
            int rheight = height(root.right); 
              
            /* use the larger one */
            if (lheight > rheight) 
                return(lheight+1); 
            else return(rheight+1);  
        } 
    } 
```

### Level of a Node in a Binary tree

```
 /* Helper function for getLevel().  It returns level of the data 
    if data is present in tree, otherwise returns 0.*/
    int getLevelUtil(Node node, int data, int level)  
    { 
        if (node == null) 
            return 0; 
   
        if (node.data == data) 
            return level; 
   
        int downlevel = getLevelUtil(node.left, data, level + 1); 
        if (downlevel != 0) 
            return downlevel; 
   
        downlevel = getLevelUtil(node.right, data, level + 1); 
        return downlevel; 
    } 
   
    /* Returns level of given data value */
    int getLevel(Node node, int data)  
    { 
        return getLevelUtil(node, data, 1); 
    } 
```

### Mirror image of a tree

```
mirror(Node root){
 if(root!=null){
   mirror(root.left);
   mirror(root.right);
   Node temp = root.left;
   root.left = root.right;
   root.right= temp;
   
 }else{
 return
 }
}
```

### Tiny URL

https://blog.kamranali.in/system-design/url-shortner

### House coloring 1

```
public int minCost(int[][] costs) {
    if(costs==null||costs.length==0)
        return 0;
 
    for(int i=1; i<costs.length; i++){
        costs[i][0] += Math.min(costs[i-1][1], costs[i-1][2]);
        costs[i][1] += Math.min(costs[i-1][0], costs[i-1][2]);
        costs[i][2] += Math.min(costs[i-1][0], costs[i-1][1]);
    }
 
    int m = costs.length-1;
    return Math.min(Math.min(costs[m][0], costs[m][1]), costs[m][2]);
}
```

### Trie

```
public class Trie {

    private class TrieNode {
        Map<Character, TrieNode> children;
        boolean endOfWord;
        public TrieNode() {
            children = new HashMap<>();
            endOfWord = false;
        }
    }

    private final TrieNode root;
    public Trie() {
        root = new TrieNode();
    }

    /**
     * Iterative implementation of insert into trie
     */
    public void insert(String word) {
        TrieNode current = root;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            TrieNode node = current.children.get(ch);
            if (node == null) {
                node = new TrieNode();
                current.children.put(ch, node);
            }
            current = node;
        }
        //mark the current nodes endOfWord as true
        current.endOfWord = true;
    }

    /**
     * Recursive implementation of insert into trie
     */
    public void insertRecursive(String word) {
        insertRecursive(root, word, 0);
    }


    private void insertRecursive(TrieNode current, String word, int index) {
        if (index == word.length()) {
            //if end of word is reached then mark endOfWord as true on current node
            current.endOfWord = true;
            return;
        }
        char ch = word.charAt(index);
        TrieNode node = current.children.get(ch);

        //if node does not exists in map then create one and put it into map
        if (node == null) {
            node = new TrieNode();
            current.children.put(ch, node);
        }
        insertRecursive(node, word, index + 1);
    }

    /**
     * Iterative implementation of search into trie.
     */
    public boolean search(String word) {
        TrieNode current = root;
        for (int i = 0; i < word.length(); i++) {
            char ch = word.charAt(i);
            TrieNode node = current.children.get(ch);
            //if node does not exist for given char then return false
            if (node == null) {
                return false;
            }
            current = node;
        }
        //return true of current's endOfWord is true else return false.
        return current.endOfWord;
    }

    /**
     * Recursive implementation of search into trie.
     */
    public boolean searchRecursive(String word) {
        return searchRecursive(root, word, 0);
    }
    private boolean searchRecursive(TrieNode current, String word, int index) {
        if (index == word.length()) {
            //return true of current's endOfWord is true else return false.
            return current.endOfWord;
        }
        char ch = word.charAt(index);
        TrieNode node = current.children.get(ch);
        //if node does not exist for given char then return false
        if (node == null) {
            return false;
        }
        return searchRecursive(node, word, index + 1);
    }

    /**
     * Delete word from trie.
     */
    public void delete(String word) {
        delete(root, word, 0);
    }

    /**
     * Returns true if parent should delete the mapping
     */
    private boolean delete(TrieNode current, String word, int index) {
        if (index == word.length()) {
            //when end of word is reached only delete if currrent.endOfWord is true.
            if (!current.endOfWord) {
                return false;
            }
            current.endOfWord = false;
            //if current has no other mapping then return true
            return current.children.size() == 0;
        }
        char ch = word.charAt(index);
        TrieNode node = current.children.get(ch);
        if (node == null) {
            return false;
        }
        boolean shouldDeleteCurrentNode = delete(node, word, index + 1);

        //if true is returned then delete the mapping of character and trienode reference from map.
        if (shouldDeleteCurrentNode) {
            current.children.remove(ch);
            //return true if no mappings are left in the map.
            return current.children.size() == 0;
        }
        return false;
    }
}
```

### Power of x 

```
int power(int x, unsigned int y) 
{ 
    int temp; 
    if( y == 0) 
        return 1; 
    temp = power(x, y/2); 
    if (y%2 == 0) 
        return temp*temp; 
    else
        return x*temp*temp; 
}
```

### Power of x negative

```
static float power(float x, int y) 
    { 
        float temp; 
        if( y == 0) 
            return 1; 
        temp = power(x, y/2);  
          
        if (y%2 == 0) 
            return temp*temp; 
        else
        { 
            if(y > 0) 
                return x * temp * temp; 
            else
                return (temp * temp) / x; 
        } 
    }
```

### Stack using one Queue

```
private LinkedList<Integer> q1 = new LinkedList<>();

// Push element x onto stack.
public void push(int x) {
    q1.add(x);
    int sz = q1.size();
    while (sz > 1) {
        q1.add(q1.remove());
        sz--;
    }
}
```

Using two queues

```
public void pop() {
    while (q1.size() > 1) {
        top = q1.remove();
        q2.add(top);
    }
    q1.remove();
    Queue<Integer> temp = q1;
    q1 = q2;
    q2 = temp;
}
```

## Buying and selling stock

<img width="839" alt="screen shot 2018-11-15 at 5 43 08 pm" src="https://user-images.githubusercontent.com/5608772/48552245-fc2d5780-e8fd-11e8-85fb-16c93dadd1e3.png">

<img width="874" alt="screen shot 2018-11-15 at 5 40 41 pm" src="https://user-images.githubusercontent.com/5608772/48552130-9f31a180-e8fd-11e8-9a83-d96b798d3e64.png">

## Snake and ladder min moves

```
static class QueueEntry {
		int vertex; // Vertex number
		int distance; // distance of this vertex from source
	};

	static int MinimumDiceThrows(int board[], int N) {

		boolean[] visited = new boolean[N];

		for (int i = 0; i < N; i++) {
			visited[i] = false;
		}

		Queue<QueueEntry> q = new LinkedList<QueueEntry>();

		visited[0] = true;
		QueueEntry s = new QueueEntry();
		s.distance = 0;
		s.vertex = 0;
		q.add(s);

		QueueEntry qe = new QueueEntry();

		while (!q.isEmpty()) {
			
			qe = q.peek();
			int vertex = qe.vertex;

			if (vertex == N - 1){
				break;
			}
				
			q.remove();
			for (int i = vertex + 1; i <= (vertex + 6) && i < N; ++i) {

				if (visited[i] == false) {

					QueueEntry cell = new QueueEntry();
					cell.distance = (qe.distance + 1);
					visited[i] = true;
					if (board[i] != -1){
						
						cell.vertex = board[i];
					
					}else{
						
						cell.vertex = i;
					
					}
					q.add(cell);
					
				}
			}
		}

		return qe.distance;
	}
```

### Longest consicutive subsequence

<img width="698" alt="screen shot 2018-12-04 at 2 53 16 pm" src="https://user-images.githubusercontent.com/5608772/49431880-6e45dd80-f7d4-11e8-95fd-367327d1d40f.png">


### Pallindrome in Likekelist

```
boolean isPalindrome(Node head)  
    { 
        slow_ptr = head; fast_ptr = head; 
        Node prev_of_slow_ptr = head; 
        Node midnode = null;  // To handle odd size list 
        boolean res = true; // initialize result 
  
        if (head != null && head.next != null)  
        { 
            /* Get the middle of the list. Move slow_ptr by 1 
               and fast_ptrr by 2, slow_ptr will have the middle 
               node */
            while (fast_ptr != null && fast_ptr.next != null)  
            { 
                fast_ptr = fast_ptr.next.next; 
  
                /*We need previous of the slow_ptr for 
                  linked lists  with odd elements */
                prev_of_slow_ptr = slow_ptr; 
                slow_ptr = slow_ptr.next; 
            } 
  
            /* fast_ptr would become NULL when there are even elements  
               in the list and not NULL for odd elements. We need to skip   
               the middle node for odd case and store it somewhere so that 
               we can restore the original list */
            if (fast_ptr != null)  
            { 
                midnode = slow_ptr; 
                slow_ptr = slow_ptr.next; 
            } 
  
            // Now reverse the second half and compare it with first half 
            second_half = slow_ptr; 
            prev_of_slow_ptr.next = null; // NULL terminate first half 
            reverse();  // Reverse the second half 
            res = compareLists(head, second_half); // compare 
  
            /* Construct the original list back */
            reverse(); // Reverse the second half again 
              
            if (midnode != null)  
            { 
                // If there was a mid node (odd size case) which                                                          
                // was not part of either first half or second half. 
                prev_of_slow_ptr.next = midnode; 
                midnode.next = second_half; 
            } else
                prev_of_slow_ptr.next = second_half; 
        } 
        return res; 
    } 
  
    /* Function to reverse the linked list  Note that this 
       function may change the head */
    void reverse()  
    { 
        Node prev = null; 
        Node current = second_half; 
        Node next; 
        while (current != null)  
        { 
            next = current.next; 
            current.next = prev; 
            prev = current; 
            current = next; 
        } 
        second_half = prev; 
    } 
  
    /* Function to check if two input lists have same data*/
    boolean compareLists(Node head1, Node head2)  
    { 
        Node temp1 = head1; 
        Node temp2 = head2; 
  
        while (temp1 != null && temp2 != null)  
        { 
            if (temp1.data == temp2.data)  
            { 
                temp1 = temp1.next; 
                temp2 = temp2.next; 
            } else 
                return false; 
        } 
  
        /* Both are empty reurn 1*/
        if (temp1 == null && temp2 == null) 
            return true; 
  
        /* Will reach here when one is NULL 
           and other is not */
        return false; 
    } 
```

### Left View and Right View of a Tree

```
// function to print right view of binary tree
    private static void printRightView(Node root)
    {
        if (root == null)
            return;

        Queue<Node> queue = new LinkedList<Node>();
        queue.add(root);

        while (!queue.isEmpty())
        {
            // number of nodes at current level
            int n = queue.size();

            // Traverse all nodes of current level
            for (int i = 1; i <= n; i++) {
                Node temp = queue.poll();

                // Print the left most element at
                // the level
                if (i == n)
                    System.out.print(temp.data + " ");

                // Add left node to queue
                if (temp.left != null)
                    queue.add(temp.left);

                // Add right node to queue
                if (temp.right != null)
                    queue.add(temp.right);
            }
        }
    }
    
    private static void printLeftView(Node root)
    {
        if (root == null)
            return;

        Queue<Node> queue = new LinkedList<Node>();
        queue.add(root);

        while (!queue.isEmpty())
        {
            // number of nodes at current level
            int n = queue.size();

            // Traverse all nodes of current level
            for (int i = 1; i <= n; i++) {
                Node temp = queue.poll();

                // Print the left most element at
                // the level
                if (i == 1)
                    System.out.print(temp.data + " ");

                // Add left node to queue
                if (temp.left != null)
                    queue.add(temp.left);

                // Add right node to queue
                if (temp.right != null)
                    queue.add(temp.right);
            }
        }
    }
```

### Top view of Binary Tree

```
private void TopView(Node root) { 
        class QueueObj { 
            Node node; 
            int hd; 
  
            QueueObj(Node node, int hd) { 
                this.node = node; 
                this.hd = hd; 
            } 
        } 
        Queue<QueueObj> q = new LinkedList<QueueObj>(); 
        Map<Integer, Node> topViewMap = new TreeMap<Integer, Node>(); 
  
        if (root == null) { 
            return; 
        } else { 
            q.add(new QueueObj(root, 0)); 
        } 
  
        System.out.println("The top view of the tree is : "); 
          
        // count function returns 1 if the container  
        // contains an element whose key is equivalent  
        // to hd, or returns zero otherwise. 
        while (!q.isEmpty()) { 
            QueueObj tmpNode = q.poll(); 
            if (!topViewMap.containsKey(tmpNode.hd)) { 
                topViewMap.put(tmpNode.hd, tmpNode.node); 
            } 
  
            if (tmpNode.node.left != null) { 
                q.add(new QueueObj(tmpNode.node.left, tmpNode.hd - 1)); 
            } 
            if (tmpNode.node.right != null) { 
                q.add(new QueueObj(tmpNode.node.right, tmpNode.hd + 1)); 
            } 
  
        } 
        for (Entry<Integer, Node> entry : topViewMap.entrySet()) { 
            System.out.print(entry.getValue().data); 
        } 
    } 
```

### Check is A tree is subtree of a binary tree

```
boolean areIdentical(Node root1, Node root2)  
    { 
   
        /* base cases */
        if (root1 == null && root2 == null) 
            return true; 
   
        if (root1 == null || root2 == null) 
            return false; 
   
        /* Check if the data of both roots is same and data of left and right 
           subtrees are also same */
        return (root1.data == root2.data 
                && areIdentical(root1.left, root2.left) 
                && areIdentical(root1.right, root2.right)); 
    } 
   
    /* This function returns true if S is a subtree of T, otherwise false */
    boolean isSubtree(Node T, Node S)  
    { 
        /* base cases */
        if (S == null)  
            return true; 
   
        if (T == null) 
            return false; 
   
        /* Check the tree with root as current node */
        if (areIdentical(T, S))  
            return true; 
   
        /* If the tree with root as current node doesn't match then 
           try left and right subtrees one by one */
        return isSubtree(T.left, S) 
                || isSubtree(T.right, S); 
    } 
```

### Shortest Path BFS Maze

```
// M x N matrix
	private static final int M = 10;
	private static final int N = 10;

	// Below arrays details all 4 possible movements from a cell
	private static final int row[] = { -1, 0, 0, 1 };
	private static final int col[] = { 0, -1, 1, 0 };

	// Function to check if it is possible to go to position (row, col)
	// from current position. The function returns false if (row, col)
	// is not a valid position or has value 0 or it is already visited
	private static boolean isValid(int mat[][], boolean visited[][],
													int row, int col)
	{
		return (row >= 0) && (row < M) && (col >= 0) && (col < N)
					   && mat[row][col] == 1 && !visited[row][col];
	}

	int BFS(int mat[][], Point src, Point dest) {

	// check source and destination cell
	// of the matrix have value 1
	if ((mat[src.x][src.y] == 0) || (mat[dest.x][dest.y] == 0))
	    return Integer.MAX_VALUE;

	boolean[][] visited = new boolean[ROW][COL];

	// Mark the source cell as visited
	visited[src.x][src.y] = true;

	// Create a queue for BFS  --> see http://stackoverflow.com/questions/11149707/best-implementation-of-java-queue
	Queue<QueueNode> q = new ArrayDeque<QueueNode>();

	// distance of source cell is 0
	QueueNode s = new QueueNode(src, 0);
	q.add(s); // Enqueue source cell

	// Do a BFS starting from source cell
	while (!q.isEmpty()) {
	    QueueNode curr = q.peek();
	    Point pt = curr.pt;

	    // If we have reached the destination cell,
	    // we are done
	    if (pt.x == dest.x && pt.y == dest.y)
		return curr.dist;

	    // Otherwise dequeue the front cell in the queue
	    // and enqueue its adjacent cells
	    q.poll();

	    for (int i = 0; i < 4; i++) {
		int row = pt.x + rowNum[i];
		int col = pt.y + colNum[i];

		// if adjacent cell is valid, has path and
		// not visited yet, enqueue it.
		if ((isValid(row, col) && mat[row][col] == 1)
			&& !visited[row][col]) {
		    // mark cell as visited and enqueue it
		    visited[row][col] = true;
		    QueueNode adjCell = new QueueNode(new Point(row, col),
			    curr.dist + 1);
		    q.add(adjCell);
		}
	    }
	}

	// return -1 if destination cannot be reached
	return Integer.MAX_VALUE;
    }
```

### Soduko backtracking

```
bool SolveSudoku(int grid[N][N])
{
    int row, col;

    // If there is no unassigned location, we are done
    if (!FindUnassignedLocation(grid, row, col))
       return true; // success!

    // consider digits 1 to 9
    for (int num = 1; num <= N; num++)
    {
        // if looks promising
        if (isSafe(grid, row, col, num))
        {
            // make tentative assignment
            grid[row][col] = num;

            // return, if success, yay!
            if (SolveSudoku(grid))
                return true;

            // failure, unmake & try again
            grid[row][col] = UNASSIGNED;
        }
    }
    return false; // this triggers backtracking
}

/* Searches the grid to find an entry that is still unassigned. If
   found, the reference parameters row, col will be set the location
   that is unassigned, and true is returned. If no unassigned entries
   remain, false is returned. */
bool FindUnassignedLocation(int grid[N][N], int &row, int &col)
{
    for (row = 0; row < N; row++)
        for (col = 0; col < N; col++)
            if (grid[row][col] == UNASSIGNED)
                return true;
    return false;
}

/* Returns a boolean which indicates whether any assigned entry
   in the specified row matches the given number. */
bool UsedInRow(int grid[N][N], int row, int num)
{
    for (int col = 0; col < N; col++)
        if (grid[row][col] == num)
            return true;
    return false;
}

/* Returns a boolean which indicates whether any assigned entry
   in the specified column matches the given number. */
bool UsedInCol(int grid[N][N], int col, int num)
{
    for (int row = 0; row < N; row++)
        if (grid[row][col] == num)
            return true;
    return false;
}

/* Returns a boolean which indicates whether any assigned entry
   within the specified 3x3 box matches the given number. */
bool UsedInBox(int grid[N][N], int boxStartRow, int boxStartCol, int num)
{
    for (int row = 0; row < SQN; row++)
        for (int col = 0; col < SQN; col++)
            if (grid[row+boxStartRow][col+boxStartCol] == num)
                return true;
    return false;
}

/* Returns a boolean which indicates whether it will be legal to assign
   num to the given row,col location. */
bool isSafe(int grid[N][N], int row, int col, int num)
{
    /* Check if 'num' is not already placed in current row,
       current column and current 3x3 box */
    return !UsedInRow(grid, row, num) &&
           !UsedInCol(grid, col, num) &&
           !UsedInBox(grid, row - row%SQN , col - col%SQN, num);
}
```

## Count no of Islands

```
static final int ROW = 5, COL = 5; 
  
    // A function to check if a given cell (row, col) can 
    // be included in DFS 
    boolean isSafe(int M[][], int row, int col, 
                   boolean visited[][]) 
    { 
        // row number is in range, column number is in range 
        // and value is 1 and not yet visited 
        return (row >= 0) && (row < ROW) && 
               (col >= 0) && (col < COL) && 
               (M[row][col]==1 && !visited[row][col]); 
    } 
  
    // A utility function to do DFS for a 2D boolean matrix. 
    // It only considers the 8 neighbors as adjacent vertices 
    void DFS(int M[][], int row, int col, boolean visited[][]) 
    { 
        // These arrays are used to get row and column numbers 
        // of 8 neighbors of a given cell 
        int rowNbr[] = new int[] {-1, -1, -1,  0, 0,  1, 1, 1}; 
        int colNbr[] = new int[] {-1,  0,  1, -1, 1, -1, 0, 1}; 
  
        // Mark this cell as visited 
        visited[row][col] = true; 
  
        // Recur for all connected neighbours 
        for (int k = 0; k < 8; ++k) 
            if (isSafe(M, row + rowNbr[k], col + colNbr[k], visited) ) 
                DFS(M, row + rowNbr[k], col + colNbr[k], visited); 
    } 
  
    // The main function that returns count of islands in a given 
    //  boolean 2D matrix 
    int countIslands(int M[][]) 
    { 
        // Make a bool array to mark visited cells. 
        // Initially all cells are unvisited 
        boolean visited[][] = new boolean[ROW][COL]; 
  
  
        // Initialize count as 0 and travese through the all cells 
        // of given matrix 
        int count = 0; 
        for (int i = 0; i < ROW; ++i) 
            for (int j = 0; j < COL; ++j) 
                if (M[i][j]==1 && !visited[i][j]) // If a cell with 
                {                                 // value 1 is not 
                    // visited yet, then new island found, Visit all 
                    // cells in this island and increment island count 
                    DFS(M, i, j, visited); 
                    ++count; 
                } 
  
        return count; 
    } 
```

## Count no of Islands Stack ## FloodFill
```
 Stack<Pair> stack = new Stack<>();
            Set<Pair> visited = new HashSet<>();

            // Starting point.
            Pair start = new Pair(2, 2);
            stack.add(start);

            while (!stack.isEmpty()) {
                Pair pair = stack.pop();
                if (matrix[pair.row][pair.col] == 3) {
                    continue;
                }
                for (int i = 0; i < directions.length; i++) {
                    int x = pair.row + directions[i][0];
                    int y = pair.col + directions[i][1];
                    if (x < 0 || x >= matrix.length || y < 0 || y >= matrix[0].length || matrix[x][y] != 2 || visited.contains(new Pair(x, y))) {
                        continue;
                    }
                    stack.add(new Pair(x, y));
                }
                visited.add(pair);
                matrix[pair.row][pair.col] = 3;
            }

```
### Min Stack

```
Stack<Integer> stack=new Stack<>();
int min=Integer.MAX_VALUE;
public void push(int x) {
    if(x<=min) {stack.push(min); min=x;}
    stack.push(x);
}
public void pop() {
    if(stack.peek()==min){ stack.pop(); min=stack.pop(); }
    else stack.pop();
}
public int top() {
    return stack.peek();
}
public int getMin() {
    return min;
}
```

### All anagram in a string

```
static void search(String pat, String txt) 
    { 
        int M = pat.length(); 
        int N = txt.length(); 
  
        // countP[]:  Store count of all  
        // characters of pattern 
        // countTW[]: Store count of current 
        // window of text 
        char[] countP = new char[MAX]; 
        char[] countTW = new char[MAX]; 
        for (int i = 0; i < M; i++) 
        { 
            (countP[pat.charAt(i)])++; 
            (countTW[txt.charAt(i)])++; 
        } 
  
        // Traverse through remaining characters 
        // of pattern 
        for (int i = M; i < N; i++) 
        { 
            // Compare counts of current window 
            // of text with counts of pattern[] 
            if (compare(countP, countTW)) 
                System.out.println("Found at Index " + 
                                          (i - M)); 
              
            // Add current character to current  
            // window 
            (countTW[txt.charAt(i)])++; 
  
            // Remove the first character of previous 
            // window 
            countTW[txt.charAt(i-M)]--; 
        } 
  
        // Check for the last window in text 
        if (compare(countP, countTW)) 
            System.out.println("Found at Index " +  
                                       (N - M)); 
    } 
```

### Zero Sum

```
static Boolean subArrayExists(int arr[]) 
    { 
        // Creates an empty hashMap hM 
        HashMap<Integer, Integer> hM =  
                        new HashMap<Integer, Integer>(); 
          
        // Initialize sum of elements 
        int sum = 0;      
          
        // Traverse through the given array 
        for (int i = 0; i < arr.length; i++) 
        {  
            // Add current element to sum 
            sum += arr[i]; 
              
            // Return true in following cases 
            // a) Current element is 0 
            // b) sum of elements from 0 to i is 0 
            // c) sum is already present in hash map 
            if (arr[i] == 0 || sum == 0 || hM.get(sum) != null)                          
                return true; 
              
            // Add sum to hash map 
            hM.put(sum, i); 
        }  
          
        // We reach here only when there is 
        // no subarray with 0 sum 
        return false; 
    } 
```
### Staircase and Fibnoci series

```
static int countWaysUtil(int n, int m) 
    { 
        int res[] = new int[n]; 
        res[0] = 1; res[1] = 1; 
        for (int i=2; i<n; i++) 
        { 
           res[i] = 0; 
           for (int j=1; j<=m && j<=i; j++) 
             res[i] += res[i-j]; 
        } 
        return res[n-1]; 
    } 
       
    // Returns number of ways to reach s'th stair 
    static int countWays(int s, int m) 
    { 
        return countWaysUtil(s+1, m); 
    } 
```

### House robbery/loot

```
int maxLoot(int *hval, int n) 
{ 
    if (n == 0) 
        return 0; 
    if (n == 1) 
        return hval[0]; 
    if (n == 2) 
        return max(hval[0], hval[1]); 
  
    // dp[i] represent the maximum value stolen 
    // so far after reaching house i. 
    int dp[n]; 
  
    // Initialize the dp[0] and dp[1] 
    dp[0] = hval[0]; 
    dp[1] = max(hval[0], hval[1]); 
  
    // Fill remaining positions 
    for (int i = 2; i<n; i++) 
        dp[i] = max(hval[i]+dp[i-2], dp[i-1]); 
  
    return dp[n-1]; 
} 
```

### Zig Zag tree

```
void printZigZagTraversal() { 
      
    // if null then return 
    if (rootNode == null) { 
    return; 
    } 
  
    // declare two stacks 
    Stack<Node> currentLevel = new Stack<>(); 
    Stack<Node> nextLevel = new Stack<>(); 
  
    // push the root 
    currentLevel.push(rootNode); 
    boolean leftToRight = true; 
  
    // check if stack is empty 
    while (!currentLevel.isEmpty()) { 
  
    // pop out of stack 
    Node node = currentLevel.pop(); 
      
    // print the data in it 
    System.out.print(node.data + " "); 
  
    // store data according to current 
    // order. 
    if (leftToRight) { 
        if (node.leftChild != null) { 
        nextLevel.push(node.leftChild); 
        } 
          
        if (node.rightChild != null) { 
        nextLevel.push(node.rightChild); 
        } 
    } 
    else { 
        if (node.rightChild != null) { 
        nextLevel.push(node.rightChild); 
        } 
          
        if (node.leftChild != null) { 
        nextLevel.push(node.leftChild); 
        } 
    } 
  
    if (currentLevel.isEmpty()) { 
        leftToRight = !leftToRight; 
        Stack<Node> temp = currentLevel; 
        currentLevel = nextLevel; 
        nextLevel = temp; 
    } 
    } 
} 
}
```

### Print Spiral tree
```
void printSpiral(struct node *root) 
{ 
    if (root == NULL)  return;   // NULL check 
  
    // Create two stacks to store alternate levels 
    stack<struct node*> s1;  // For levels to be printed from right to left 
    stack<struct node*> s2;  // For levels to be printed from left to right 
  
    // Push first level to first stack 's1' 
    s1.push(root); 
  
    // Keep ptinting while any of the stacks has some nodes 
    while (!s1.empty() || !s2.empty()) 
    { 
        // Print nodes of current level from s1 and push nodes of 
        // next level to s2 
        while (!s1.empty()) 
        { 
            struct node *temp = s1.top(); 
            s1.pop(); 
            cout << temp->data << " "; 
  
            // Note that is right is pushed before left 
            if (temp->right) 
                s2.push(temp->right); 
            if (temp->left) 
                s2.push(temp->left); 
        } 
  
        // Print nodes of current level from s2 and push nodes of 
        // next level to s1 
        while (!s2.empty()) 
        { 
            struct node *temp = s2.top(); 
            s2.pop(); 
            cout << temp->data << " "; 
  
            // Note that is left is pushed before right 
            if (temp->left) 
                s1.push(temp->left); 
            if (temp->right) 
                s1.push(temp->right); 
        } 
    } 
} 
```

### Frog Jump

https://www.youtube.com/watch?v=jH_5ypQggWg&t=614s

```
public boolean canCross(int[] stones) {
        int n = stones.length;
        int[] dp = new int[n];
        for (int i = 0; i < n; i++) {
            dp[i] = Integer.MAX_VALUE;
        }
        dp[0] = 0;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (i <= j + stones[j]) {
                    dp[i] = Math.min(dp[i], dp[j] + 1);
                }
            }
        }
        return dp[n - 1] == -1 ? false : true;
    }

```
### Fence Painting Algorithm

```
for n = 1
    diff = k, same = 0
    total = k

for n = 2
    diff = k * (k-1) //k choices for
           first post, k-1 for next
    same = k //k choices for common 
           color of two posts
    total = k +  k * (k-1)

for n = 3
    diff = [k +  k * (k-1)] * (k-1) 
           (k-1) choices for 3rd post 
           to not have color of 2nd 
           post.
    same = k * (k-1) 
           c'' != c, (k-1) choices for it

Hence we deduce that,
total[i] = same[i] + diff[i]
same[i]  = diff[i-1]
diff[i]  = (diff[i-1] + diff[i-2]) * (k-1)
         = total[i-1] * (k-1)
	 
	 
 long countWays(int n, int k) 
{ 
    // To store results for subproblems 
    long dp[n + 1]; 
    // There are k ways to color first post 
    dp[1] = k; 
  
    // There are 0 ways for single post to 
    // violate (same color_ and k ways to 
    // not violate (different color) 
    int same = 0, diff = k; 
  
    // Fill for 2 posts onwards 
    for (int i = 2; i <= n; i++) 
    { 
        // Current same is same as previous diff 
        same = diff; 
  
        // We always have k-1 choices for next post 
        diff = dp[i-1] * (k-1); 
       
        // Total choices till i. 
        dp[i] = (same + diff); 
    } 
  
    return dp[n]; 
}
```

### Check if two trees re identical

```
boolean identicalTrees(Node a, Node b)  
    { 
        /*1. both empty */
        if (a == null && b == null) 
            return true; 
              
        /* 2. both non-empty -> compare them */
        if (a != null && b != null)  
            return (a.data == b.data 
                    && identicalTrees(a.left, b.left) 
                    && identicalTrees(a.right, b.right)); 
   
        /* 3. one empty, one not -> false */
        return false; 
    } 
```
### 0,1,2 seggregation

```
void segg(int[] arr){
   int low=0,high=arr.length,mid=0;
   while(mid<=high){
     switch(a[mid])
       Case 0:
         Swap(a[low],a[mid]);
	 low++,mid++;
	 break;
	Case 1:
	  mid++;
	  break;
	Case 2:
	  Swap(a[mid],a[high]);
	  high--;
	  break;
   }
}
```

### Autocomplete using trie

```
private class Node {
    String prefix;
    HashMap<Character, Node> children;
        
    // Does this node represent the last character in a word?
    boolean isWord;
        
    private Node(String prefix) {
        this.prefix = prefix;
        this.children = new HashMap<Character, Node>();
    }
}
    
// The trie
private Node trie;
    
// Construct the trie from the dictionary
public Autocomplete(String[] dict) {
    trie = new Node("");
    for (String s : dict) insertWord(s);
}
    
// Insert a word into the trie
private void insertWord(String s) {
    // Iterate through each character in the string. If the character is not
    // already in the trie then add it
    Node curr = trie;
    for (int i = 0; i < s.length(); i++) {
        if (!curr.children.containsKey(s.charAt(i))) {
            curr.children.put(s.charAt(i), new Node(s.substring(0, i + 1)));
        }
        curr = curr.children.get(s.charAt(i));
        if (i == s.length() - 1) curr.isWord = true;
    }
}
    
// Find all words in trie that start with prefix
public List<String> getWordsForPrefix(String pre) {
    List<String> results = new LinkedList<String>();
        
    // Iterate to the end of the prefix
    Node curr = trie;
    for (char c : pre.toCharArray()) {
        if (curr.children.containsKey(c)) {
            curr = curr.children.get(c);
        } else {
            return results;
        }
    }
        
    // At the end of the prefix, find all child words
    findAllChildWords(curr, results);
    return results;
}
    
// Recursively find every child word
private void findAllChildWords(Node n, List<String> results) {
    if (n.isWord) results.add(n.prefix);
    for (Character c : n.children.keySet()) {
        findAllChildWords(n.children.get(c), results);
    }
}
```

### Overlaping rectangles

```
bool doOverlap(Point l1, Point r1, Point l2, Point r2) 
{ 
    // If one rectangle is on left side of other 
    if (l1.x > r2.x || l2.x > r1.x) 
        return false; 
  
    // If one rectangle is above other 
    if (l1.y < r2.y || l2.y < r1.y) 
        return false; 
  
    return true; 
} 
```

### Max rectangle in an matrix of 1

```
public int maximum(int input[][]){
        int temp[] = new int[input[0].length];
        MaximumHistogram mh = new MaximumHistogram();
        int maxArea = 0;
        int area = 0;
        for(int i=0; i < input.length; i++){
            for(int j=0; j < temp.length; j++){
                if(input[i][j] == 0){
                    temp[j] = 0;
                }else{
                    temp[j] += input[i][j];
                }
            }
            area = mh.maxHistogram(temp);
            if(area > maxArea){
                maxArea = area;
            }
        }
        return maxArea;
    }
```

### Max Histogram area

```
 public int maxHistogram(int input[]){
        Deque<Integer> stack = new LinkedList<Integer>();
        int maxArea = 0;
        int area = 0;
        int i;
        for(i=0; i < input.length;){
            if(stack.isEmpty() || input[stack.peekFirst()] <= input[i]){
                stack.offerFirst(i++);
            }else{
                int top = stack.pollFirst();
                //if stack is empty means everything till i has to be
                //greater or equal to input[top] so get area by
                //input[top] * i;
                if(stack.isEmpty()){
                    area = input[top] * i;
                }
                //if stack is not empty then everythin from i-1 to input.peek() + 1
                //has to be greater or equal to input[top]
                //so area = input[top]*(i - stack.peek() - 1);
                else{
                    area = input[top] * (i - stack.peekFirst() - 1);
                }
                if(area > maxArea){
                    maxArea = area;
                }
            }
        }
        while(!stack.isEmpty()){
            int top = stack.pollFirst();
            //if stack is empty means everything till i has to be
            //greater or equal to input[top] so get area by
            //input[top] * i;
            if(stack.isEmpty()){
                area = input[top] * i;
            }
            //if stack is not empty then everything from i-1 to input.peek() + 1
            //has to be greater or equal to input[top]
            //so area = input[top]*(i - stack.peek() - 1);
            else{
                area = input[top] * (i - stack.peekFirst() - 1);
            }
        if(area > maxArea){
                maxArea = area;
            }
        }
        return maxArea;
    }
```
### Command line pattern Undo redo

```
http://www.newthinktank.com/2012/09/command-design-pattern-tutorial/
```

### Word break
```
bool wordBreak(string str) 
{ 
    int size = str.size(); 
  
    // Base case 
    if (size == 0)  return true; 
  
    // Try all prefixes of lengths from 1 to size 
    for (int i=1; i<=size; i++) 
    { 
        // The parameter for dictionaryContains is  
        // str.substr(0, i) which is prefix (of input  
        // string) of length 'i'. We first check whether  
        // current prefix is in  dictionary. Then we  
        // recursively check for remaining string 
        // str.substr(i, size-i) which is suffix of   
        // length size-i 
        if (dictionaryContains( str.substr(0, i) ) && 
            wordBreak( str.substr(i, size-i) )) 
            return true; 
    } 
  
    // If we have tried all prefixes and  
    // none of them worked 
    return false; 
} 
```

### Check Loop in a linkelist
```
private boolean checkLoop(Node head){
        Node p=head;
        Node q=head;
        while(p!=null && q!=null && q.next!=null){
            p=p.next;
            q=q.next.next;
            if(p==q){
                return true;
            }
        }
        return false;
    }
```

### Sort an array or stack using stack
```
stack<int> sortStack(stack<int> input) 
{ 
    stack<int> tmpStack; 
  
    while (!input.empty()) 
    { 
        // pop out the first element 
        int tmp = input.top(); 
        input.pop(); 
  
        // while temporary stack is not empty 
        // and top of stack is smaller than temp 
        while (!tmpStack.empty() && 
                tmpStack.top() < tmp) 
        { 
            // pop from temporary stack and 
            // push it to the input stack 
            input.push(tmpStack.top()); 
            tmpStack.pop(); 
        } 
  
        // push temp in tempory of stack 
        tmpStack.push(tmp); 
    } 
  
    return tmpStack; 
} 
  
void sortArrayUsingStacks(int arr[], int n) 
{ 
    // Push array elements to stack 
    stack<int> input; 
    for (int i=0; i<n; i++) 
       input.push(arr[i]); 
  
    // Sort the temporary stack 
    stack<int> tmpStack = sortStack(input); 
  
    // Put stack elements in arrp[] 
    for (int i=0; i<n; i++) 
    { 
        arr[i] = tmpStack.top(); 
        tmpStack.pop(); 
    } 
} 
```

### Pages tree - linking of pages is a website is good
```
# breadth first
good(pages, visitedPages, depth)
  if depth &gt; 5
    return false
  if pages.empty
    return true
  newPages = List()
  for p in pages
    subPages = f(p)
    for sp in subPages
      if !visitedPages.contains(sp)
        visitedPages.add(sp)
        newPages.add(sp)
  return good(newPages, visitedPages, depth + 1)

good(homepage, List(), 0)
```

### Two nodes of a BST are swapped, correct the BST

<img width="942" alt="screen shot 2018-12-05 at 12 36 05 am" src="https://user-images.githubusercontent.com/5608772/49466352-d375ef00-f825-11e8-914c-0f791e8e9cee.png">

### Given a sorted dictionary of an alien language, find order of characters

```
class Graph 
{ 
  
    // An array representing the graph as an adjacency list 
    private final LinkedList<Integer>[] adjacencyList; 
  
    Graph(int nVertices) 
    { 
        adjacencyList = new LinkedList[nVertices]; 
        for (int vertexIndex = 0; vertexIndex < nVertices; vertexIndex++) 
        { 
            adjacencyList[vertexIndex] = new LinkedList<>(); 
        } 
    } 
  
    // function to add an edge to graph 
    void addEdge(int startVertex, int endVertex) 
    { 
        adjacencyList[startVertex].add(endVertex); 
    } 
  
    private int getNoOfVertices() 
    { 
        return adjacencyList.length; 
    } 
  
    // A recursive function used by topologicalSort 
    private void topologicalSortUtil(int currentVertex, boolean[] visited, 
                                     Stack<Integer> stack) 
    { 
        // Mark the current node as visited. 
        visited[currentVertex] = true; 
  
        // Recur for all the vertices adjacent to this vertex 
        for (int adjacentVertex : adjacencyList[currentVertex]) 
        { 
            if (!visited[adjacentVertex]) 
            { 
                topologicalSortUtil(adjacentVertex, visited, stack); 
            } 
        } 
  
        // Push current vertex to stack which stores result 
        stack.push(currentVertex); 
    } 
  
    // prints a Topological Sort of the complete graph 
    void topologicalSort() 
    { 
        Stack<Integer> stack = new Stack<>(); 
  
        // Mark all the vertices as not visited 
        boolean[] visited = new boolean[getNoOfVertices()]; 
        for (int i = 0; i < getNoOfVertices(); i++) 
        { 
            visited[i] = false; 
        } 
  
        // Call the recursive helper function to store Topological  
        // Sort starting from all vertices one by one 
        for (int i = 0; i < getNoOfVertices(); i++) 
        { 
            if (!visited[i]) 
            { 
                topologicalSortUtil(i, visited, stack); 
            } 
        } 
  
        // Print contents of stack 
        while (!stack.isEmpty()) 
        { 
            System.out.print((char)('a' + stack.pop()) + " "); 
        } 
    } 
} 
  
public class OrderOfCharacters 
{ 
    // This function fidns and prints order 
    // of characer from a sorted array of words. 
    // alpha is number of possible alphabets  
    // starting from 'a'. For simplicity, this 
    // function is written in a way that only 
    // first 'alpha' characters can be there  
    // in words array. For example if alpha 
    //  is 7, then words[] should contain words 
    // having only 'a', 'b','c' 'd', 'e', 'f', 'g' 
    private static void printOrder(String[] words, int alpha) 
    { 
        // Create a graph with 'aplha' edges 
        Graph graph = new Graph(alpha); 
  
        for (int i = 0; i < words.length - 1; i++) 
        { 
            // Take the current two words and find the first mismatching 
            // character 
            String word1 = words[i]; 
            String word2 = words[i+1]; 
            for (int j = 0; j < Math.min(word1.length(), word2.length()); j++) 
            { 
                // If we find a mismatching character, then add an edge 
                // from character of word1 to that of word2 
                if (word1.charAt(j) != word2.charAt(j)) 
                { 
                    graph.addEdge(word1.charAt(j) - 'a', word2.charAt(j)- 'a'); 
                    break; 
                } 
            } 
        } 
  
        // Print topological sort of the above created graph 
        graph.topologicalSort(); 
    } 
  
    // Driver program to test above functions 
    public static void main(String[] args) 
    { 
        String[] words = {"caa", "aaa", "aab"}; 
        printOrder(words, 3); 
    } 
} 
```
### Intersection of two linkelist

```
int getIntesectionNode(struct Node* head1, struct Node* head2) 
{ 
  int c1 = getCount(head1); 
  int c2 = getCount(head2); 
  int d; 
  
  if(c1 > c2) 
  { 
    d = c1 - c2; 
    return _getIntesectionNode(d, head1, head2); 
  } 
  else
  { 
    d = c2 - c1; 
    return _getIntesectionNode(d, head2, head1); 
  } 
} 
  
/* function to get the intersection point of two linked 
   lists head1 and head2 where head1 has d more nodes than 
   head2 */
int _getIntesectionNode(int d, struct Node* head1, struct Node* head2) 
{ 
  int i; 
  struct Node* current1 = head1; 
  struct Node* current2 = head2; 
  
  for(i = 0; i < d; i++) 
  { 
    if(current1 == NULL) 
    {  return -1; } 
    current1 = current1->next; 
  } 
  
  while(current1 !=  NULL && current2 != NULL) 
  { 
    if(current1 == current2) 
      return current1->data; 
    current1= current1->next; 
    current2= current2->next; 
  } 
  
  return -1; 
} 
  
/* Takes head pointer of the linked list and 
   returns the count of nodes in the list */
int getCount(struct Node* head) 
{ 
  struct Node* current = head; 
  int count = 0; 
  
  while (current != NULL) 
  { 
    count++; 
    current = current->next; 
  } 
  
  return count; 
} 
```

### Triplet in an array
```
boolean find3Numbers(int A[], int arr_size, int sum) 
    { 
        int l, r; 
  
        /* Sort the elements */
        quickSort(A, 0, arr_size - 1); 
  
        /* Now fix the first element one by one and find the 
           other two elements */
        for (int i = 0; i < arr_size - 2; i++) { 
  
            // To find the other two elements, start two index variables 
            // from two corners of the array and move them toward each 
            // other 
            l = i + 1; // index of the first element in the remaining elements 
            r = arr_size - 1; // index of the last element 
            while (l < r) { 
                if (A[i] + A[l] + A[r] == sum) { 
                    System.out.print("Triplet is " + A[i] + 
                                 ", " + A[l] + ", " + A[r]); 
                    return true; 
                } 
                else if (A[i] + A[l] + A[r] < sum) 
                    l++; 
  
                else // A[i] + A[l] + A[r] > sum 
                    r--; 
            } 
        } 
  
        // If we reach here, then no triplet was found 
        return false; 
    } 
```

### Find Pair with given sum in Binary search tree

```
bool findpairUtil(Node* root, int sum,  unordered_set<int> &set) 
{ 
    if (root == NULL) 
        return false; 
  
    if (findpairUtil(root->left, sum, set)) 
        return true; 
  
    if (set.find(sum - root->data) != set.end()) { 
        cout << "Pair is found ("
             << sum - root->data << ", "
             << root->data << ")" << endl; 
        return true; 
    } 
    else
        set.insert(root->data); 
  
    return findpairUtil(root->right, sum, set); 
} 
```


### Add one ot 1 to linkelist number
```
Node *addOneUtil(Node *head) 
{ 
    // res is head node of the resultant list 
    Node* res = head; 
    Node *temp, *prev = NULL; 
  
    int carry = 1, sum; 
  
    while (head != NULL) //while both lists exist 
    { 
        // Calculate value of next digit in resultant list. 
        // The next digit is sum of following things 
        // (i) Carry 
        // (ii) Next digit of head list (if there is a 
        //     next digit) 
        sum = carry + head->data; 
  
        // update carry for next calulation 
        carry = (sum >= 10)? 1 : 0; 
  
        // update sum if it is greater than 10 
        sum = sum % 10; 
  
        // Create a new node with sum as data 
        head->data = sum; 
  
        // Move head and second pointers to next nodes 
        temp = head; 
        head = head->next; 
    } 
  
    // if some carry is still there, add a new node to 
    // result list. 
    if (carry > 0) 
        temp->next = newNode(carry); 
  
    // return head of the resultant list 
    return res; 
} 
```

### Sort stack using recursion
```
void sortedInsert(struct stack **s, int x) 
{ 
    // Base case: Either stack is empty or newly inserted 
    // item is greater than top (more than all existing) 
    if (isEmpty(*s) || x > top(*s)) 
    { 
        push(s, x); 
        return; 
    } 
  
    // If top is greater, remove the top item and recur 
    int temp = pop(s); 
    sortedInsert(s, x); 
  
    // Put back the top item removed earlier 
    push(s, temp); 
} 
  
// Function to sort stack 
void sortStack(struct stack **s) 
{ 
    // If stack is not empty 
    if (!isEmpty(*s)) 
    { 
        // Remove the top item 
        int x = pop(s); 
  
        // Sort remaining stack 
        sortStack(s); 
  
        // Push the top item back in sorted stack 
        sortedInsert(s, x); 
    } 
} 
```

### Find no of employees under an employee
```
// This function populates 'result' for given input 'dataset' 
    private static void populateResult(Map<String, String> dataSet) 
    { 
        // To store reverse of original map, each key will have 0 
        // to multiple values 
        Map<String, List<String>> mngrEmpMap = 
                                  new HashMap<String, List<String>>(); 
  
        // To fill mngrEmpMap, iterate through the given map 
        for (Map.Entry<String,String> entry: dataSet.entrySet()) 
        { 
            String emp = entry.getKey(); 
            String mngr = entry.getValue(); 
            if (!emp.equals(mngr)) // excluding emp-emp entry 
            { 
                // Get the previous list of direct reports under 
                // current 'mgr' and add the current 'emp' to the list 
                List<String> directReportList = mngrEmpMap.get(mngr); 
  
                // If 'emp' is the first employee under 'mgr' 
                if (directReportList == null) 
                    directReportList = new ArrayList<String>(); 
  
                directReportList.add(emp); 
                  
                // Replace old value for 'mgr' with new 
                // directReportList 
                mngrEmpMap.put(mngr, directReportList); 
            } 
        } 
  
        // Now use manager-Emp map built above to populate result  
        // with use of populateResultUtil() 
  
        // note- we are iterating over original emp-manager map and 
        // will use mngr-emp map in helper to get the count 
        for (String mngr: dataSet.keySet()) 
            populateResultUtil(mngr, mngrEmpMap); 
    } 
  
    // This is a recursive function to fill count for 'mgr' using 
    // mngrEmpMap.  This function uses memoization to avoid re- 
    // computations of subproblems. 
    private static int populateResultUtil(String mngr, 
                               Map<String, List<String>> mngrEmpMap) 
    { 
        int count = 0; 
  
        // means employee is not a manager of any other employee 
        if (!mngrEmpMap.containsKey(mngr)) 
        { 
            result.put(mngr, 0); 
            return 0; 
        } 
  
        // this employee count has already been done by this 
        // method, so avoid re-computation 
        else if (result.containsKey(mngr)) 
            count = result.get(mngr); 
  
        else
        { 
            List<String> directReportEmpList = mngrEmpMap.get(mngr); 
            count = directReportEmpList.size(); 
            for (String directReportEmp: directReportEmpList) 
               count +=  populateResultUtil(directReportEmp, mngrEmpMap); 
  
            result.put(mngr, count); 
        } 
        return count; 
    } 
} 
```

### Is Tree BST

```
boolean isBST()  { 
        return isBSTUtil(root, Integer.MIN_VALUE, 
                               Integer.MAX_VALUE); 
    } 
  
    /* Returns true if the given tree is a BST and its 
      values are >= min and <= max. */
    boolean isBSTUtil(Node node, int min, int max) 
    { 
        /* an empty tree is BST */
        if (node == null) 
            return true; 
  
        /* false if this node violates the min/max constraints */
        if (node.data < min || node.data > max) 
            return false; 
  
        /* otherwise check the subtrees recursively 
        tightening the min/max constraints */
        // Allow only distinct values 
        return (isBSTUtil(node.left, min, node.data-1) && 
                isBSTUtil(node.right, node.data+1, max)); 
    } 
```
### Diagonal view of Tree

```
static void diagonalPrintUtil(Node root,int d, 
            HashMap<Integer,Vector<Integer>> diagonalPrint){ 
          
         // Base case 
        if (root == null) 
            return; 
          
        // get the list at the particular d value 
        Vector<Integer> k = diagonalPrint.get(d); 
          
        // k is null then create a vector and store the data 
        if (k == null) 
        { 
            k = new Vector<>(); 
            k.add(root.data); 
        } 
          
        // k is not null then update the list 
        else
        { 
            k.add(root.data); 
        } 
          
        // Store all nodes of same line together as a vector 
        diagonalPrint.put(d,k); 
          
        // Increase the vertical distance if left child 
        diagonalPrintUtil(root.left, d + 1, diagonalPrint); 
           
        // Vertical distance remains same for right child 
        diagonalPrintUtil(root.right, d, diagonalPrint); 
    } 
      
    // Print diagonal traversal of given binary tree 
    static void diagonalPrint(Node root) 
    { 
        // create a map of vectors to store Diagonal elements 
        HashMap<Integer,Vector<Integer>> diagonalPrint = new HashMap<>(); 
        diagonalPrintUtil(root, 0, diagonalPrint); 
          
        System.out.println("Diagonal Traversal of Binnary Tree"); 
        for (Entry<Integer, Vector<Integer>> entry : diagonalPrint.entrySet()) 
        { 
            System.out.println(entry.getValue()); 
        } 
    } 
```

### Sum of k elements in BST

```
int ksmallestElementSumRec(Node *root, int k, int &count) 
{ 
    // Base cases 
    if (root == NULL) 
        return 0; 
    if (count > k) 
        return 0; 
  
    // Compute sum of elements in left subtree 
    int res = ksmallestElementSumRec(root->left, k, count); 
    if (count >= k) 
        return res; 
  
    // Add root's data 
    res += root->data; 
  
    // Add current Node 
    count++; 
    if (count >= k) 
      return res; 
  
    // If count is less than k, return right subtree Nodes 
    return res + ksmallestElementSumRec(root->right, k, count); 
} 
```

### Kth largest element in BST

```
void kthLargestUtil(Node *root, int k, int &c) 
{ 
    // Base cases, the second condition is important to 
    // avoid unnecessary recursive calls 
    if (root == NULL || c >= k) 
        return; 
  
    // Follow reverse inorder traversal so that the 
    // largest element is visited first 
    kthLargestUtil(root->right, k, c); 
  
    // Increment count of visited nodes 
    c++; 
  
    // If c becomes k now, then this is the k'th largest  
    if (c == k) 
    { 
        cout << "K'th largest element is "
             << root->key << endl; 
        return; 
    } 
  
    // Recur for left subtree 
    kthLargestUtil(root->left, k, c); 
} 
```
### Path to given sum.

```
int sum=0;
int givenSum = 20;
Stack<Integer> s;
void printPath(Node root){
  if(root==null) return;
  sum = sum +root.data;
  s.push(root.data);
  if(sum==givenSum)
    print stack
  printPath(root.left);
  printPath(root.right);
  sum = sum -root.data;
  s.pop();
}
```

### Print all paths of a binary tree

```
Stack<Integer> s;
printAllPath(Node root){
 if(root==null) return;
  s.push(root.data);
  printPath(root.left);
  if(root.left==null && root.right==null)
    print stack
  printPath(root.right);
  s.pop();
}
```

### Path exists in a binary tree

```
bool existPath(struct Node *root, int arr[], int n, int index) 
{ 
    // If root is NULL, then there must not be any element 
    // in array. 
    if (root == NULL) 
        return (n == 0); 
  
   // If this node is a leaf and matches with last entry 
   // of array. 
   if ((root->left == NULL && root->right == NULL) && 
       (root->data == arr[index]) && (index == n-1)) 
            return true; 
  
   // If current node is equal to arr[index] this means 
   // that till this level path has been matched and 
   // remaining path can be either in left subtree or 
   // right subtree. 
   return ((index < n) && (root->data == arr[index]) && 
              (existPath(root->left, arr, n,  index+1) || 
               existPath(root->right, arr, n, index+1) )); 
}
```

### Cousins of a node in a tree

```
void printGivenLevel(Node* root, Node *node, int level) 
{ 
    // Base cases 
    if (root == NULL || level < 2) 
        return; 
  
    // If current node is parent of a node with 
    // given level 
    if (level == 2) 
    { 
        if (root->left == node || root->right == node) 
            return; 
        if (root->left) 
           printf("%d ", root->left->data); 
        if (root->right) 
           printf("%d ", root->right->data); 
    } 
  
    // Recur for left and right subtrees 
    else if (level > 2) 
    { 
        printGivenLevel(root->left, node, level-1); 
        printGivenLevel(root->right, node, level-1); 
    } 
} 
  
// This function prints cousins of a given node 
void printCousins(Node *root, Node *node) 
{ 
    // Get level of given node 
    int level = getLevel(root, node, 1); 
  
    // Print nodes of given level. 
    printGivenLevel(root, node, level); 
}
```

### Print nodes k distance of the tree
```
void printKDistant(Node node, int k)  
    { 
        if (node == null) 
            return; 
        if (k == 0)  
        { 
            System.out.print(node.data + " "); 
            return; 
        }  
        else 
        { 
            printKDistant(node.left, k - 1); 
            printKDistant(node.right, k - 1); 
        } 
    } 
```
### form Max num from a number

```
static int printMaxNum(int num) 
    { 
        // hashed array to store count of digits 
        int count[] = new int[10]; 
          
        // Converting given number to string 
        String str = Integer.toString(num); 
          
        // Updating the count array 
        for(int i=0; i < str.length(); i++) 
            count[str.charAt(i)-'0']++; 
          
        // result is to store the final number 
        int result = 0, multiplier = 1; 
          
        // Traversing the count array 
        // to calculate the maximum number 
        for (int i = 0; i <= 9; i++) 
        { 
            while (count[i] > 0) 
            { 
                result = result + (i * multiplier); 
                count[i]--; 
                multiplier = multiplier * 10; 
            } 
        } 
       
        // return the result 
        return result; 
    } 
```

### Form the largest number using at most one swap operation 

```
static String largestNumber(String num) 
    { 
        int n = num.length(); 
        int right; 
        int rightMax[] = new int[n]; 
  
        // for the rightmost digit, there 
        // will be no greater right digit 
        rightMax[n - 1] = -1; 
  
        // index of the greatest right digit 
        // till the current index from the 
        // right direction 
        right = n - 1; 
  
        // traverse the array from second right 
        // element up to the left element 
        for (int i = n - 1; i >= 0 ; i--) 
        { 
            // if 'num.charAt(i)' is less than the 
            // greatest digit encountered so far 
            if (num.charAt(i) < num.charAt(right)) 
                rightMax[i] = right; 
  
            else
            { 
                // there is no greater right digit 
                // for 'num.charAt(i)' 
                rightMax[i] = -1; 
  
                // update 'right' index 
                right = i; 
            } 
        } 
  
        // traverse the 'rightMax[]' array from 
        // left to right 
        for (int i = 0; i < n; i++) 
        { 
  
            // if for the current digit, greater 
            // right digit exists then swap it 
            // with its greater right digit and break 
            if (rightMax[i] != -1) 
            { 
                // performing the required swap operation 
                num = swap(num,i,rightMax[i]); 
                break; 
            } 
        } 
  
        // required largest number 
        return num; 
    } 
```
### Form the smallest number using at most one swap operation

```
 public static String smallestNumber(String str){ 
          
        char[] num = str.toCharArray(); 
        int n = str.length(); 
        int[] rightMin = new int[n]; 
  
        // for the rightmost digit, there 
        // will be no smaller right digit 
        rightMin[n - 1] = -1; 
  
        // index of the smallest right digit  
        // till the current index from the  
        // right direction 
        int right = n - 1; 
  
        // traverse the array from second  
        // right element up to the left  
        // element 
        for (int i = n - 2; i >= 1; i--)  
        {  
            // if 'num[i]' is greater than  
            // the smallest digit  
            // encountered so far 
            if (num[i] > num[right]) 
            rightMin[i] = right; 
  
            else
            {  
            // there is no smaller right  
            // digit for 'num[i]' 
            rightMin[i] = -1; 
  
            // update 'right' index 
            right = i; 
            } 
        } 
  
        // special condition for the 1st  
        // digit so that it is not swapped  
        // with digit '0' 
        int small = -1; 
        for (int i = 1; i < n; i++) 
            if (num[i] != '0') 
            { 
                if (small == -1) 
                { 
                    if (num[i] < num[0]) 
                        small = i; 
                } 
                else if (num[i] < num[small]) 
                    small = i;                  
            } 
      
        if (small != -1){ 
            char temp; 
            temp = num[0]; 
            num[0] = num[small]; 
            num[small] = temp; 
        } 
        else
        { 
            // traverse the 'rightMin[]'  
            // array from 2nd digit up  
            // to the last digit 
            for (int i = 1; i < n; i++)  
            {  
                // if for the current digit,  
                // smaller right digit exists,  
                // then swap it with its smaller 
                // right digit and break 
                if (rightMin[i] != -1)  
                {  
                    // performing the required  
                    // swap operation 
                    char temp; 
                    temp = num[i]; 
                    num[i] = num[rightMin[i]]; 
                    num[rightMin[i]] = temp; 
                    break; 
                } 
            } 
        } 
  
        // required smallest number 
        return (new String(num));          
    } 
      
    // driver function 
    public static void main(String argc[]){ 
        String num = "9625635"; 
        System.out.println("Smallest number: "+ 
                          smallestNumber(num)); 
    } 
} 
```

### Check if Robot moves are cicular

```
static boolean isCircular(char path[]) 
{ 
  // Initialize starting 
  // point for robot as  
  // (0, 0) and starting 
  // direction as N North 
  int x = 0, y = 0; 
  int dir = 0; 
   
  // Traverse the path given for robot 
  for (int i=0; i < path.length; i++) 
  { 
      // Find current move 
      char move = path[i]; 
   
      // If move is left or 
      // right, then change direction 
      if (move == 'R') 
        dir = (dir + 1)%4; 
      else if (move == 'L') 
        dir = (4 + dir - 1) % 4; 
   
      // If move is Go, then  
      // change  x or y according to 
      // current direction 
      else // if (move == 'G') 
      { 
         if (dir == 0) 
            y++; 
         else if (dir == 1) 
            x++; 
         else if (dir == 2) 
            y--; 
         else // dir == 3 
            x--; 
      } 
  } 
   
   // If robot comes back to 
   // (0, 0), then path is cyclic 
  return (x == 0 && y == 0); 
} 
```

### Merge K sorted arrays

```
private class QueueNode implements Comparable<QueueNode> {
    int array, index, value;
 
 
    public QueueNode(int array, int index, int value) {
        this.array = array;
        this.index = index;
        this.value = value;
    }
 
    public int compareTo(QueueNode n) {
        if (value > n.value) return 1;
        if (value < n.value) return -1;
        return 0;
    }
}
 
public int[] merge(int[][] arrays) {
    PriorityQueue<QueueNode> pq = new PriorityQueue<QueueNode>();
 
    int size = 0;
    for (int i = 0; i < arrays.length; i++) {
        size += arrays[i].length;
        if (arrays[i].length > 0) {
            pq.add(new QueueNode(i, 0, arrays[i][0]);
        }
    }
 
    int[] result = new int[size];
    for (int i = 0; !pq.isEmpty(); i++) {
        QueueNode n = pq.poll();
        result[i] = n.value;
        int newIndex = n.index + 1;
        if (newIndex < arrays[n.array].length) {
            pq.add(new QueueNode(n.array, newIndex, 
            arrays[n.array][newIndex]);
        }
    }
 
    return result;
}
```

### Subarray with sum k

```
int subArraySum(int arr[], int n, int sum)  
    { 
        int curr_sum = arr[0], start = 0, i; 
  
        // Pick a starting point 
        for (i = 1; i <= n; i++)  
        { 
            // If curr_sum exceeds the sum, then remove the starting elements 
            while (curr_sum > sum && start < i-1) 
            { 
                curr_sum = curr_sum - arr[start]; 
                start++; 
            } 
              
            // If curr_sum becomes equal to sum, then return true 
            if (curr_sum == sum)  
            { 
                int p = i-1; 
                System.out.println("Sum found between indexes " + start 
                        + " and " + p); 
                return 1; 
            } 
              
            // Add this element to curr_sum 
            if (i < n) 
            curr_sum = curr_sum + arr[i]; 
              
        } 
  
        System.out.println("No subarray found"); 
        return 0; 
    } 
  
    public static void main(String[] args)  
    { 
        SubarraySum arraysum = new SubarraySum(); 
        int arr[] = {15, 2, 4, 8, 9, 5, 10, 23}; 
        int n = arr.length; 
        int sum = 23; 
        arraysum.subArraySum(arr, n, sum); 
    } 
} 
```

### Subarray with sum k negative
```
public static void subArraySum(int[] arr, int n, int sum) { 
        //cur_sum to keep track of cummulative sum till that point 
        int cur_sum = 0; 
        int start = 0; 
        int end = -1; 
        HashMap<Integer, Integer> hashMap = new HashMap<>(); 
  
        for (int i = 0; i < n; i++) { 
            cur_sum = cur_sum + arr[i]; 
            //check whether cur_sum - sum = 0, if 0 it means 
            //the sub array is starting from index 0- so stop 
            if (cur_sum - sum == 0) { 
                start = 0; 
                end = i; 
                break; 
            } 
            //if hashMap already has the value, means we already  
            // have subarray with the sum - so stop 
            if (hashMap.containsKey(cur_sum - sum)) { 
                start = hashMap.get(cur_sum - sum) + 1; 
                end = i; 
                break; 
            } 
            //if value is not present then add to hashmap 
            hashMap.put(cur_sum, i); 
  
        } 
        // if end is -1 : means we have reached end without the sum 
        if (end == -1) { 
            System.out.println("No subarray with given sum exists"); 
        } else { 
            System.out.println("Sum found between indexes " 
                            + start + " to " + end); 
        } 
  
    } 
```

### Top view and bottom view of tree

```
 private void TopView(Node root) { 
        class QueueObj { 
            Node node; 
            int hd; 
  
            QueueObj(Node node, int hd) { 
                this.node = node; 
                this.hd = hd; 
            } 
        } 
        Queue<QueueObj> q = new LinkedList<QueueObj>(); 
        Map<Integer, Node> topViewMap = new TreeMap<Integer, Node>(); 
  
        if (root == null) { 
            return; 
        } else { 
            q.add(new QueueObj(root, 0)); 
        } 
  
        System.out.println("The top view of the tree is : "); 
          
        // count function returns 1 if the container  
        // contains an element whose key is equivalent  
        // to hd, or returns zero otherwise. 
        while (!q.isEmpty()) { 
            QueueObj tmpNode = q.poll(); 
	    // In case of bottom view , don't check directly override
            if (!topViewMap.containsKey(tmpNode.hd)) { 
                topViewMap.put(tmpNode.hd, tmpNode.node); 
            } 
  
            if (tmpNode.node.left != null) { 
                q.add(new QueueObj(tmpNode.node.left, tmpNode.hd - 1)); 
            } 
            if (tmpNode.node.right != null) { 
                q.add(new QueueObj(tmpNode.node.right, tmpNode.hd + 1)); 
            } 
  
        } 
        for (Entry<Integer, Node> entry : topViewMap.entrySet()) { 
            System.out.print(entry.getValue().data); 
        } 
    } 
```
### Subarray sum of size K

```
int subArraySum(int arr[], int n, int sum)  
    { 
        int curr_sum = arr[0], start = 0, i; 
  
        // Pick a starting point 
        for (i = 1; i <= n; i++)  
        { 
            // If curr_sum exceeds the sum, then remove the starting elements 
            while (curr_sum > sum && start < i-1) 
            { 
                curr_sum = curr_sum - arr[start]; 
                start++; 
            } 
              
            // If curr_sum becomes equal to sum, then return true 
            if (curr_sum == sum)  
            { 
                int p = i-1; 
                System.out.println("Sum found between indexes " + start 
                        + " and " + p); 
                return 1; 
            } 
              
            // Add this element to curr_sum 
            if (i < n) 
            curr_sum = curr_sum + arr[i]; 
              
        } 
  
        System.out.println("No subarray found"); 
        return 0; 
    } 
```

### Subarray sum of size k negative

```
unordered_map<int, int> map; 
  
    // Maintains sum of elements so far 
    int curr_sum = 0; 
  
    for (int i = 0; i < n; i++) 
    { 
        // add current element to curr_sum 
        curr_sum = curr_sum + arr[i]; 
  
        // if curr_sum is equal to target sum 
        // we found a subarray starting from index 0 
        // and ending at index i 
        if (curr_sum == sum) 
        { 
            cout << "Sum found between indexes "
                 << 0 << " to " << i << endl; 
            return; 
        } 
  
        // If curr_sum - sum already exists in map 
        // we have found a subarray with target sum 
        if (map.find(curr_sum - sum) != map.end()) 
        { 
            cout << "Sum found between indexes "
                 << map[curr_sum - sum] + 1 
                 << " to " << i << endl; 
            return; 
        } 
  
        map[curr_sum] = i; 
    } 
  
```
### State machine

```
class Transition {
    State from;
    Set<Condition> conditions;
    State to;
}

class State {
    String state;
}

class Condition {
    String condition;
}

class StateMachine {
    List<Transition> transitions;
    State current;

    StateMachine(State start, List<Transition> transitions) {
        this.current = start;
        this.transitions = transitions;
    }

    void apply(Set<Condition> conditions) {
        current = getNextState(conditions);
    }

    State getNextState(Set<Condition> conditions) {
        for(Transition transition : transitions) {
            boolean currentStateMatches = transition.from.equals(current);
            boolean conditionsMatch = transition.conditions.equals(conditions);
            if(currentStateMatches && conditionsMatch) {
                return transition.to;
            }
        }
        return null;
    }
}


State one = new State("one");
State two = new State("two");
State three = new State("three");

Condition sunday = new Condition("Sunday");
Condition raining = new Condition("Raining");
Condition notSunday = new Condition("Not Sunday");
Condition notRaining = new Condition("Not Raining");

List<Transition> transitions = new ArrayList<Transition>();
transitions.add(one, new Set(sunday), three);
transitions.add(one, new Set(sunday), two); // <<--- Invalid, cant go to two and three
transitions.add(one, new Set(raining), three);
transitions.add(one, new Set(sunday, raining), three);
transitions.add(one, new Set(notSunday, notRaining), three);

StateMachine machine = new StateMachine(one, transitions);
System.out.print(machine.current); // "one"
machine.apply(new Set(sunday, raining));
System.out.print(machine.current); // "three
```

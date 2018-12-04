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

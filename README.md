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


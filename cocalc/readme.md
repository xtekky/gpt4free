### Example: `cocalc` <a name="example-cocalc"></a>


```python
# import library
import cocalc

cocalc.Completion.create(prompt="How are you!", cookieInput="cookieinput") ## Tutorial 
```

### How to grab cookie input
```js
// input this into ur developer tools console and the exact response u get from this u put into ur cookieInput!
var cookies = document.cookie.split("; ");
var cookieString = "";
for (var i = 0; i < cookies.length; i++) {
  cookieString += cookies[i] + "; ";
}
console.log(cookieString);
```

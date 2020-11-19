
var x = new Array[2];
x[0] = new Array[2];
x[0][0] = 11
x[0][1] = 22
var y = [[0, 1], [2, 3]]
var z = JSON.parse(JSON.stringify(y[0]))
console.log(z)
let img;
let arr=[];
let posx;
let posy;
let dir;
let steps=250000;
aUP=0;
aRIGHT=1;
aDOWN=2;
aLEFT=3;
function setup() {
    createCanvas(400, 400);
    colorMode(HSB);
    img  = createImage(width,height);
    for (let i = 0; i < width; i++) {
        arr[i]=[];
        for (let j = 0; j < height; j++) {
            arr[i][j]={val:0,visits:0};
        }
    }
    posx=width/2;
    posy=width/2;
    
    dir=aUP;
}
  
function turnRight(){
    dir++;
    if(dir>aLEFT){
        dir =aUP;
    }
}

function turnLeft(){
    dir--;
    if(dir<aUP){
        dir =aLEFT;
    }
}

function moveForward(){
    if(dir==aUP){
        posx--;
    }
    if(dir==aRIGHT){
        posy++;
    }
    if(dir==aDOWN){
        posx++;
    }
    if(dir==aLEFT){
        posy--;
    }

    if(posx>=width){
        posx=0;
    }
    if(posx<0){
        posx=width-1;
    }
    if(posy>=height){
        posy=0;
    }
    if(posy<0){
        posy=height-1;
    }


}
function flipSquare(){
    if(arr[posx][posy]['val']==0){
        arr[posx][posy]['val']=1;
        arr[posx][posy]['visits']=(arr[posx][posy]['visits']+1)%360;

    }
    else if(arr[posx][posy]['val']==1){arr[posx][posy]['val']=0;}
    
}

function draw() {
    background(0);
    
    
    for(let i =0;i<steps;i++){
        if(arr[posx][posy]['val']==0){
            turnRight();
        }
        if(arr[posx][posy]['val']==1){
            turnLeft();
        }
        flipSquare();      
        moveForward();
    }

    img.loadPixels();
    for (let i = 0; i < img.width; i++) {
        for (let j = 0; j < img.height; j++) {
        c=255;
        
        if(arr[i][j]['visits']>0){ 
            
            c=arr[i][j]['visits']%360;
        };
          img.set(i, j, color(c,100,100));
        }
    }

    img.updatePixels();
    image(img,0,0)
}
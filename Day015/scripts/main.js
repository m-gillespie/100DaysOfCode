let lines =[];
let w;
let index;
let offset;
let a;
let r;

function setup() {
    createCanvas(1200, 1200);
    index =-1;
    offset=0;
    w=1;
    a=0;
    r=height*.5;
    cx=width/2;
    cy=height/2;
    for(let i =0;i<width;i++){
        lines[i]=random(0,360)
    }
    colorMode(HSB);
}
  
function swap(i){
    a = lines[i];
    b = lines[i+1];

    if(a<b){
        lines[i+1]=a;
        lines[i]=b;
    }

}

function draw() {
    background(0);
    // push();
    // translate(cx,cy)    


    // for(let j =0;j<5000;j++){
    //     swap(index)
    //     index++;
    //     if(index>=lines.length-offset){index=0;offset++;}
    // }



    // slices=TWO_PI/lines.length;
    // for(let i=0;i<lines.length;i++){

    //     // console.log(i)
    //     noStroke();
        
    //     colorMode(HSB);
    //     fill(lines[i],100,90)
    //     // console.log(i*slices)
    //     arc(0,0,800,800,i*slices,i*slices+slices)        


    // }
    // pop();
    // Lines
    for(let j = 0; j<10000;j++){
        swap(index)
        index++;
        if(index>=lines.length-offset){index=0;offset++;}
    }

    noStroke()
    for(let i =0;i<lines.length;i++){
        
        fill(lines[i],100,100);

        // if(i>lines.length-offset-1){fill(0,255,0);}
        // if(i == index){fill(255,0,0);}
        h = map(lines[i],0,360,0,height)
        rect(i*w,h,w,height)
    }

    if(offset>lines.length){
        noLoop();
    }
    

}
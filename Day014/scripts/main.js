let cirs =[];
let dir;
let max = 5000;
let stuckC = 0;
function setup() {
    createCanvas(1200, 800);
    
    cirs.push(new Particle(300,height-5,5));
    cirs.push(new Particle(600,height-5,5));
    cirs.push(new Particle(900,height-5,5));
    cirs[0].stuck=true;
    cirs[1].stuck=true;
    cirs[2].stuck=true;
    for(let i = 0;i<1;i++){
        cirs.push(new Particle(this.random(width),random(-100,0),2))
    }
    
    
  }
  
  function draw() {
    background(0);

    if(cirs.length<max){
        cirs.push(new Particle(this.random(width),random(-100,0),2))
    }

    for(cir of cirs){
        cir.randomWalk()
        cir.show()

        if(!cir.stuck){
            for(let i =0;i<cirs.length;i++){
    
                if(cirs[i].stuck){
                   isStuck =cir.checkStuck(cirs[i]);
                   if(isStuck){
                    stuckC++;
                    cir.stickTo(cirs[i]);
                    break;
                   }
                }
                
            }

        }
        
    }
    // console.log(dir)
    
    // cir = cir+dir.mult(1);
    // cir.x =cir.x+dir.x;
    // cir.y = cir.y+dir.y;

    // console.log(cir.x)
    



    circle(cir.x,cir.y,25);
    // noLoop()

  }
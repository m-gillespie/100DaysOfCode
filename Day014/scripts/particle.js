

class Particle {
    
    constructor(_x,_y,_r) {
        this.vec = createVector(_x,_y);
        this.r = _r
        this.dir = p5.Vector.random2D().mult(5);
        this.stuck = false;
        this.bias = createVector(0,3);
        this.distT = 5
        this.stuckTo = null;
        this.color=0;
    }

    randomWalk(){
        if(!this.stuck){
            this.vec.x += random(-5,5)
            this.vec.y += random(-5,5)
            this.vec.add(this.bias);

            if(this.vec.y>height){this.vec.y=0;this.vec.x=random(width)}
            if(this.vec.x<0){this.vec.x=width}
            if(this.vec.x>width){this.vec.x=0}
            // console.log(this.vec.x)
        }
        
    }
    stickTo(c){
        this.stuck=true;
        this.stuckTo = c;
        this.color = (c.color +1)%360;
    }
    checkStuck(c){

        let distance= dist(c.vec.x,c.vec.y,this.vec.x,this.vec.y);
        let target = this.r + c.r;
        return distance<=target+this.distT;
        


    }
    update(){
        if(!this.stuck){
            this.vec = this.vec.add(this.dir);

            if(this.vec.x<0||this.vec.x>width){this.dir.x=this.dir.x*-1}
            if(this.vec.y>height||this.vec.y<0){this.dir.y=this.dir.y*-1}
        }

    }

    show(){
        
        noStroke()
        fill(255)
        colorMode(HSB,360,100,100);


        if(this.stuckTo){
            stroke(this.color,100,100);
            strokeWeight(this.r)
            line(this.vec.x,this.vec.y,this.stuckTo.vec.x,this.stuckTo.vec.y);
        }
        if(this.stuck){
            fill(this.color,100,100)

        }
        noStroke();
        circle(this.vec.x,this.vec.y,this.r*2);


    }



}
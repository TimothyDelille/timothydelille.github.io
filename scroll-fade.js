/*-----SCROLLING EFFECT-----*/   
$.fn.moveIt = function(){
  var $window = $(window);
  var instances = [];
  
  $(this).each(function(){
    instances.push(new moveItItem($(this)));
  });
  
  window.addEventListener('scroll', function(){
    var scrollTop = $window.scrollTop();
    instances.forEach(function(inst){
      inst.update(scrollTop);
    });
  }, {passive: true});
}

var moveItItem = function(el){
  this.el = $(el);
};

moveItItem.prototype.update = function(scrollTop){
$decalage = -30;
if(this.el.offset().top - $(window).scrollTop() < $decalage){
    $height = $decalage;
    $opacite = 1 + (this.el.offset().top - $(window).scrollTop())/this.el.height();
}
else if(this.el.offset().top - $(window).scrollTop() > $(window).height()){
    $height = -$decalage;
}
else{
    $height = 0;
    $opacite = ($(window).scrollTop()+$(window).height()-this.el.offset().top)/this.el.height();
}
this.el.css('transform','translateY('+ $height + 'px)');
this.el.css('opacity',$opacite);
};
// Initialization
$(function(){
  $('.scroll-fade').moveIt();
});
// ne pas mettre overflow:hidden dans le .html sinon ne fonctionne pas
$('.scroll-fade').css('transition','300ms ease-out');
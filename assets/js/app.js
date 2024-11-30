(function($) {

 function init() {
    /* Sidebar height set */
    $sidebarStyles = $('.sidebar').attr('style') || "";
    $sidebarStyles += ' min-height: ' + $(document).height() + 'px;';
    $('.sidebar').attr('style', $sidebarStyles);

    /* Secondary contact links */
    var $scontacts = $('#contact-list-secondary');
    var $contactList = $('#contact-list');

    $scontacts.hide();
    $contactList.mouseenter(function(){ $scontacts.fadeIn(); });
    $contactList.mouseleave(function(){ $scontacts.fadeOut(); });

    /**
     * Tags & categories tab activation based on hash value. If hash is undefined then first tab is activated.
     */
    function activateTab() {
      if(['/tags.html', '/categories.html'].indexOf(window.location.pathname) > -1) {
        var hash = decodeURIComponent(window.location.hash);
        if(hash)
          $('.tab-pane').length && $('a[href="' + hash + '"]').tab('show');
        else
          $('.tab-pane').length && $($('.cat-tag-menu li a')[0]).tab('show');
      }
    }

    // watch hash change and activate relevant tab
    $(window).on('hashchange', activateTab);

    // initial activation
    activateTab();
  };

  // run init on document ready
  $(document).ready(init);

})(jQuery);

$(document).ready(function(){ 
  //익스 8~9 스크립트
  var trident = navigator.userAgent.match(/Trident\/(\d.\d)/i);
  if(trident != null){
    if(trident[1] < '6.0'){
      $(".ani").removeClass('ani');
    }
  }
  
  $('.s2_wrap > span').click(function() {
    $('.s2_wrap > span.on').removeClass("on");
    $(this).addClass("on");
    $('.s2_wrap > i.on').removeClass("on");
    $("."+$(this).data("button")).addClass('on');
    $(".content_box").attr('class','content_box');
    $(".content_box").addClass($(this).data("show"));
    $(".section02 .content").hide();
    $(".section02 .content_box ."+$(this).data("show")).show();
  });

  $('.s2_wrap > span').mouseenter(function(){
    var hover1 = $("."+$(this).data("button"));
    var hover2 = hover1.hasClass('on');
    if ( hover2 == false)
    { 
      $("."+$(this).data("button")).addClass('hover');
    }
  });

  $('.s2_wrap > span').mouseleave(function(){
    $("."+$(this).data("button")).removeClass('hover');
  });
  
  
});
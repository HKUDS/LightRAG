import{p as V}from"./chunk-353BL4L5-BJGenYOY.js";import{a8 as S,a3 as z,aG as U,_ as u,g as j,s as q,a as H,b as Z,q as J,p as K,l as F,c as Q,D as X,H as Y,N as tt,e as et,y as at,F as rt}from"./index-bjrbS6e8.js";import{p as nt}from"./treemap-75Q7IDZK-DNUGBdnj.js";import{d as P}from"./arc-D-vjsldI.js";import{o as it}from"./ordinal-BENe2yWM.js";import"./_baseUniq-DknB5v3H.js";import"./_basePickBy-UdMCOwSh.js";import"./clone-g5iXXiWA.js";import"./init-Gi6I4Gst.js";function st(t,a){return a<t?-1:a>t?1:a>=t?0:NaN}function ot(t){return t}function lt(){var t=ot,a=st,m=null,o=S(0),p=S(z),x=S(0);function i(e){var r,l=(e=U(e)).length,g,A,h=0,c=new Array(l),n=new Array(l),v=+o.apply(this,arguments),w=Math.min(z,Math.max(-z,p.apply(this,arguments)-v)),f,T=Math.min(Math.abs(w)/l,x.apply(this,arguments)),$=T*(w<0?-1:1),d;for(r=0;r<l;++r)(d=n[c[r]=r]=+t(e[r],r,e))>0&&(h+=d);for(a!=null?c.sort(function(y,D){return a(n[y],n[D])}):m!=null&&c.sort(function(y,D){return m(e[y],e[D])}),r=0,A=h?(w-l*$)/h:0;r<l;++r,v=f)g=c[r],d=n[g],f=v+(d>0?d*A:0)+$,n[g]={data:e[g],index:r,value:d,startAngle:v,endAngle:f,padAngle:T};return n}return i.value=function(e){return arguments.length?(t=typeof e=="function"?e:S(+e),i):t},i.sortValues=function(e){return arguments.length?(a=e,m=null,i):a},i.sort=function(e){return arguments.length?(m=e,a=null,i):m},i.startAngle=function(e){return arguments.length?(o=typeof e=="function"?e:S(+e),i):o},i.endAngle=function(e){return arguments.length?(p=typeof e=="function"?e:S(+e),i):p},i.padAngle=function(e){return arguments.length?(x=typeof e=="function"?e:S(+e),i):x},i}var ct=rt.pie,G={sections:new Map,showData:!1},b=G.sections,N=G.showData,ut=structuredClone(ct),pt=u(()=>structuredClone(ut),"getConfig"),gt=u(()=>{b=new Map,N=G.showData,at()},"clear"),dt=u(({label:t,value:a})=>{b.has(t)||(b.set(t,a),F.debug(`added new section: ${t}, with value: ${a}`))},"addSection"),ft=u(()=>b,"getSections"),mt=u(t=>{N=t},"setShowData"),ht=u(()=>N,"getShowData"),R={getConfig:pt,clear:gt,setDiagramTitle:K,getDiagramTitle:J,setAccTitle:Z,getAccTitle:H,setAccDescription:q,getAccDescription:j,addSection:dt,getSections:ft,setShowData:mt,getShowData:ht},vt=u((t,a)=>{V(t,a),a.setShowData(t.showData),t.sections.map(a.addSection)},"populateDb"),yt={parse:u(async t=>{const a=await nt("pie",t);F.debug(a),vt(a,R)},"parse")},St=u(t=>`
  .pieCircle{
    stroke: ${t.pieStrokeColor};
    stroke-width : ${t.pieStrokeWidth};
    opacity : ${t.pieOpacity};
  }
  .pieOuterCircle{
    stroke: ${t.pieOuterStrokeColor};
    stroke-width: ${t.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${t.pieTitleTextSize};
    fill: ${t.pieTitleTextColor};
    font-family: ${t.fontFamily};
  }
  .slice {
    font-family: ${t.fontFamily};
    fill: ${t.pieSectionTextColor};
    font-size:${t.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${t.pieLegendTextColor};
    font-family: ${t.fontFamily};
    font-size: ${t.pieLegendTextSize};
  }
`,"getStyles"),xt=St,At=u(t=>{const a=[...t.entries()].map(o=>({label:o[0],value:o[1]})).sort((o,p)=>p.value-o.value);return lt().value(o=>o.value)(a)},"createPieArcs"),wt=u((t,a,m,o)=>{F.debug(`rendering pie chart
`+t);const p=o.db,x=Q(),i=X(p.getConfig(),x.pie),e=40,r=18,l=4,g=450,A=g,h=Y(a),c=h.append("g");c.attr("transform","translate("+A/2+","+g/2+")");const{themeVariables:n}=x;let[v]=tt(n.pieOuterStrokeWidth);v??(v=2);const w=i.textPosition,f=Math.min(A,g)/2-e,T=P().innerRadius(0).outerRadius(f),$=P().innerRadius(f*w).outerRadius(f*w);c.append("circle").attr("cx",0).attr("cy",0).attr("r",f+v/2).attr("class","pieOuterCircle");const d=p.getSections(),y=At(d),D=[n.pie1,n.pie2,n.pie3,n.pie4,n.pie5,n.pie6,n.pie7,n.pie8,n.pie9,n.pie10,n.pie11,n.pie12],C=it(D);c.selectAll("mySlices").data(y).enter().append("path").attr("d",T).attr("fill",s=>C(s.data.label)).attr("class","pieCircle");let W=0;d.forEach(s=>{W+=s}),c.selectAll("mySlices").data(y).enter().append("text").text(s=>(s.data.value/W*100).toFixed(0)+"%").attr("transform",s=>"translate("+$.centroid(s)+")").style("text-anchor","middle").attr("class","slice"),c.append("text").text(p.getDiagramTitle()).attr("x",0).attr("y",-400/2).attr("class","pieTitleText");const M=c.selectAll(".legend").data(C.domain()).enter().append("g").attr("class","legend").attr("transform",(s,k)=>{const E=r+l,L=E*C.domain().length/2,_=12*r,B=k*E-L;return"translate("+_+","+B+")"});M.append("rect").attr("width",r).attr("height",r).style("fill",C).style("stroke",C),M.data(y).append("text").attr("x",r+l).attr("y",r-l).text(s=>{const{label:k,value:E}=s.data;return p.getShowData()?`${k} [${E}]`:k});const I=Math.max(...M.selectAll("text").nodes().map(s=>(s==null?void 0:s.getBoundingClientRect().width)??0)),O=A+e+r+l+I;h.attr("viewBox",`0 0 ${O} ${g}`),et(h,g,O,i.useMaxWidth)},"draw"),Dt={draw:wt},Gt={parser:yt,db:R,renderer:Dt,styles:xt};export{Gt as diagram};

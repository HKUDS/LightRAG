import{p as N}from"./chunk-353BL4L5-BbtbjO9C.js";import{_ as i,g as B,s as U,a as q,b as H,t as K,q as V,l as C,c as Z,F as j,K as J,M as Q,N as z,O as X,e as Y,z as tt,P as et,H as at}from"./mermaid-vendor-BWqoPkHO.js";import{p as rt}from"./treemap-75Q7IDZK-TyhSBCxN.js";import"./feature-graph-BnXy9sZT.js";import"./react-vendor-DEwriMA6.js";import"./graph-vendor-B-X5JegA.js";import"./ui-vendor-CeCm8EER.js";import"./utils-vendor-BysuhMZA.js";import"./_baseUniq-BXnQ2z3w.js";import"./_basePickBy-DmoAs6l5.js";import"./clone-BckWjv4Y.js";var it=at.pie,D={sections:new Map,showData:!1},f=D.sections,w=D.showData,st=structuredClone(it),ot=i(()=>structuredClone(st),"getConfig"),nt=i(()=>{f=new Map,w=D.showData,tt()},"clear"),lt=i(({label:t,value:a})=>{f.has(t)||(f.set(t,a),C.debug(`added new section: ${t}, with value: ${a}`))},"addSection"),ct=i(()=>f,"getSections"),pt=i(t=>{w=t},"setShowData"),dt=i(()=>w,"getShowData"),F={getConfig:ot,clear:nt,setDiagramTitle:V,getDiagramTitle:K,setAccTitle:H,getAccTitle:q,setAccDescription:U,getAccDescription:B,addSection:lt,getSections:ct,setShowData:pt,getShowData:dt},gt=i((t,a)=>{N(t,a),a.setShowData(t.showData),t.sections.map(a.addSection)},"populateDb"),ut={parse:i(async t=>{const a=await rt("pie",t);C.debug(a),gt(a,F)},"parse")},mt=i(t=>`
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
`,"getStyles"),ft=mt,ht=i(t=>{const a=[...t.entries()].map(s=>({label:s[0],value:s[1]})).sort((s,n)=>n.value-s.value);return et().value(s=>s.value)(a)},"createPieArcs"),St=i((t,a,G,s)=>{C.debug(`rendering pie chart
`+t);const n=s.db,y=Z(),T=j(n.getConfig(),y.pie),$=40,o=18,d=4,c=450,h=c,S=J(a),l=S.append("g");l.attr("transform","translate("+h/2+","+c/2+")");const{themeVariables:r}=y;let[A]=Q(r.pieOuterStrokeWidth);A??(A=2);const _=T.textPosition,g=Math.min(h,c)/2-$,M=z().innerRadius(0).outerRadius(g),O=z().innerRadius(g*_).outerRadius(g*_);l.append("circle").attr("cx",0).attr("cy",0).attr("r",g+A/2).attr("class","pieOuterCircle");const b=n.getSections(),v=ht(b),P=[r.pie1,r.pie2,r.pie3,r.pie4,r.pie5,r.pie6,r.pie7,r.pie8,r.pie9,r.pie10,r.pie11,r.pie12],p=X(P);l.selectAll("mySlices").data(v).enter().append("path").attr("d",M).attr("fill",e=>p(e.data.label)).attr("class","pieCircle");let E=0;b.forEach(e=>{E+=e}),l.selectAll("mySlices").data(v).enter().append("text").text(e=>(e.data.value/E*100).toFixed(0)+"%").attr("transform",e=>"translate("+O.centroid(e)+")").style("text-anchor","middle").attr("class","slice"),l.append("text").text(n.getDiagramTitle()).attr("x",0).attr("y",-400/2).attr("class","pieTitleText");const x=l.selectAll(".legend").data(p.domain()).enter().append("g").attr("class","legend").attr("transform",(e,u)=>{const m=o+d,R=m*p.domain().length/2,I=12*o,L=u*m-R;return"translate("+I+","+L+")"});x.append("rect").attr("width",o).attr("height",o).style("fill",p).style("stroke",p),x.data(v).append("text").attr("x",o+d).attr("y",o-d).text(e=>{const{label:u,value:m}=e.data;return n.getShowData()?`${u} [${m}]`:u});const W=Math.max(...x.selectAll("text").nodes().map(e=>(e==null?void 0:e.getBoundingClientRect().width)??0)),k=h+$+o+d+W;S.attr("viewBox",`0 0 ${k} ${c}`),Y(S,c,k,T.useMaxWidth)},"draw"),vt={draw:St},kt={parser:ut,db:F,renderer:vt,styles:ft};export{kt as diagram};

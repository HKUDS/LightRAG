import{_ as s,g as U,s as q,a as H,b as K,t as V,q as Z,l as w,c as j,F as J,K as Q,M as X,N as G,O as Y,e as ee,z as te,P as ae,H as re}from"./mermaid-vendor-B7K7CNJU.js";import{p as ie}from"./chunk-4BX2VUAB-CHD465TF.js";import{p as se}from"./treemap-75Q7IDZK-BwKiH7zL.js";import"./feature-graph-DMc8AgwX.js";import"./react-vendor-CI9PN-jl.js";import"./graph-vendor-C2ay-rh5.js";import"./ui-vendor-BNB5trKt.js";import"./utils-vendor-BaQYYNaI.js";import"./_baseUniq-43q82aez.js";import"./_basePickBy-DEs_nWf2.js";import"./clone-CFyjr8pW.js";var le=re.pie,D={sections:new Map,showData:!1},g=D.sections,C=D.showData,oe=structuredClone(le),ne=s(()=>structuredClone(oe),"getConfig"),ce=s(()=>{g=new Map,C=D.showData,te()},"clear"),pe=s(({label:e,value:a})=>{if(a<0)throw new Error(`"${e}" has invalid value: ${a}. Negative values are not allowed in pie charts. All slice values must be >= 0.`);g.has(e)||(g.set(e,a),w.debug(`added new section: ${e}, with value: ${a}`))},"addSection"),de=s(()=>g,"getSections"),ge=s(e=>{C=e},"setShowData"),ue=s(()=>C,"getShowData"),M={getConfig:ne,clear:ce,setDiagramTitle:Z,getDiagramTitle:V,setAccTitle:K,getAccTitle:H,setAccDescription:q,getAccDescription:U,addSection:pe,getSections:de,setShowData:ge,getShowData:ue},fe=s((e,a)=>{ie(e,a),a.setShowData(e.showData),e.sections.map(a.addSection)},"populateDb"),me={parse:s(async e=>{const a=await se("pie",e);w.debug(a),fe(a,M)},"parse")},he=s(e=>`
  .pieCircle{
    stroke: ${e.pieStrokeColor};
    stroke-width : ${e.pieStrokeWidth};
    opacity : ${e.pieOpacity};
  }
  .pieOuterCircle{
    stroke: ${e.pieOuterStrokeColor};
    stroke-width: ${e.pieOuterStrokeWidth};
    fill: none;
  }
  .pieTitleText {
    text-anchor: middle;
    font-size: ${e.pieTitleTextSize};
    fill: ${e.pieTitleTextColor};
    font-family: ${e.fontFamily};
  }
  .slice {
    font-family: ${e.fontFamily};
    fill: ${e.pieSectionTextColor};
    font-size:${e.pieSectionTextSize};
    // fill: white;
  }
  .legend text {
    fill: ${e.pieLegendTextColor};
    font-family: ${e.fontFamily};
    font-size: ${e.pieLegendTextSize};
  }
`,"getStyles"),ve=he,Se=s(e=>{const a=[...e.values()].reduce((r,l)=>r+l,0),$=[...e.entries()].map(([r,l])=>({label:r,value:l})).filter(r=>r.value/a*100>=1).sort((r,l)=>l.value-r.value);return ae().value(r=>r.value)($)},"createPieArcs"),xe=s((e,a,$,y)=>{w.debug(`rendering pie chart
`+e);const r=y.db,l=j(),T=J(r.getConfig(),l.pie),A=40,o=18,p=4,c=450,u=c,f=Q(a),n=f.append("g");n.attr("transform","translate("+u/2+","+c/2+")");const{themeVariables:i}=l;let[b]=X(i.pieOuterStrokeWidth);b??(b=2);const _=T.textPosition,d=Math.min(u,c)/2-A,O=G().innerRadius(0).outerRadius(d),P=G().innerRadius(d*_).outerRadius(d*_);n.append("circle").attr("cx",0).attr("cy",0).attr("r",d+b/2).attr("class","pieOuterCircle");const m=r.getSections(),W=Se(m),N=[i.pie1,i.pie2,i.pie3,i.pie4,i.pie5,i.pie6,i.pie7,i.pie8,i.pie9,i.pie10,i.pie11,i.pie12];let h=0;m.forEach(t=>{h+=t});const E=W.filter(t=>(t.data.value/h*100).toFixed(0)!=="0"),v=Y(N);n.selectAll("mySlices").data(E).enter().append("path").attr("d",O).attr("fill",t=>v(t.data.label)).attr("class","pieCircle"),n.selectAll("mySlices").data(E).enter().append("text").text(t=>(t.data.value/h*100).toFixed(0)+"%").attr("transform",t=>"translate("+P.centroid(t)+")").style("text-anchor","middle").attr("class","slice"),n.append("text").text(r.getDiagramTitle()).attr("x",0).attr("y",-400/2).attr("class","pieTitleText");const k=[...m.entries()].map(([t,x])=>({label:t,value:x})),S=n.selectAll(".legend").data(k).enter().append("g").attr("class","legend").attr("transform",(t,x)=>{const F=o+p,I=F*k.length/2,L=12*o,B=x*F-I;return"translate("+L+","+B+")"});S.append("rect").attr("width",o).attr("height",o).style("fill",t=>v(t.label)).style("stroke",t=>v(t.label)),S.append("text").attr("x",o+p).attr("y",o-p).text(t=>r.getShowData()?`${t.label} [${t.value}]`:t.label);const R=Math.max(...S.selectAll("text").nodes().map(t=>(t==null?void 0:t.getBoundingClientRect().width)??0)),z=u+A+o+p+R;f.attr("viewBox",`0 0 ${z} ${c}`),ee(f,c,z,T.useMaxWidth)},"draw"),we={draw:xe},Fe={parser:me,db:M,renderer:we,styles:ve};export{Fe as diagram};

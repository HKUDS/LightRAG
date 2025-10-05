import{p as y}from"./chunk-353BL4L5-CffmyEvg.js";import{_ as l,s as B,g as S,t as z,q as F,a as P,b as E,F as v,K as W,e as T,z as D,G as _,H as A,l as w}from"./mermaid-vendor-DB77lfsV.js";import{p as N}from"./treemap-75Q7IDZK-DV-jJKWx.js";import"./feature-graph-c3KAmzq6.js";import"./react-vendor-DEwriMA6.js";import"./graph-vendor-B-X5JegA.js";import"./ui-vendor-CeCm8EER.js";import"./utils-vendor-BysuhMZA.js";import"./_baseUniq-CAqbga58.js";import"./_basePickBy-wRSrd0Dh.js";import"./clone-DXKtE0u2.js";var x={packet:[]},m=structuredClone(x),L=A.packet,Y=l(()=>{const t=v({...L,..._().packet});return t.showBits&&(t.paddingY+=10),t},"getConfig"),G=l(()=>m.packet,"getPacket"),H=l(t=>{t.length>0&&m.packet.push(t)},"pushWord"),I=l(()=>{D(),m=structuredClone(x)},"clear"),u={pushWord:H,getPacket:G,getConfig:Y,clear:I,setAccTitle:E,getAccTitle:P,setDiagramTitle:F,getDiagramTitle:z,getAccDescription:S,setAccDescription:B},K=1e4,M=l(t=>{y(t,u);let e=-1,o=[],n=1;const{bitsPerRow:i}=u.getConfig();for(let{start:a,end:r,bits:c,label:f}of t.blocks){if(a!==void 0&&r!==void 0&&r<a)throw new Error(`Packet block ${a} - ${r} is invalid. End must be greater than start.`);if(a??(a=e+1),a!==e+1)throw new Error(`Packet block ${a} - ${r??a} is not contiguous. It should start from ${e+1}.`);if(c===0)throw new Error(`Packet block ${a} is invalid. Cannot have a zero bit field.`);for(r??(r=a+(c??1)-1),c??(c=r-a+1),e=r,w.debug(`Packet block ${a} - ${e} with label ${f}`);o.length<=i+1&&u.getPacket().length<K;){const[d,p]=O({start:a,end:r,bits:c,label:f},n,i);if(o.push(d),d.end+1===n*i&&(u.pushWord(o),o=[],n++),!p)break;({start:a,end:r,bits:c,label:f}=p)}}u.pushWord(o)},"populate"),O=l((t,e,o)=>{if(t.start===void 0)throw new Error("start should have been set during first phase");if(t.end===void 0)throw new Error("end should have been set during first phase");if(t.start>t.end)throw new Error(`Block start ${t.start} is greater than block end ${t.end}.`);if(t.end+1<=e*o)return[t,void 0];const n=e*o-1,i=e*o;return[{start:t.start,end:n,label:t.label,bits:n-t.start},{start:i,end:t.end,label:t.label,bits:t.end-i}]},"getNextFittingBlock"),q={parse:l(async t=>{const e=await N("packet",t);w.debug(e),M(e)},"parse")},R=l((t,e,o,n)=>{const i=n.db,a=i.getConfig(),{rowHeight:r,paddingY:c,bitWidth:f,bitsPerRow:d}=a,p=i.getPacket(),s=i.getDiagramTitle(),k=r+c,g=k*(p.length+1)-(s?0:r),b=f*d+2,h=W(e);h.attr("viewbox",`0 0 ${b} ${g}`),T(h,g,b,a.useMaxWidth);for(const[C,$]of p.entries())U(h,$,C,a);h.append("text").text(s).attr("x",b/2).attr("y",g-k/2).attr("dominant-baseline","middle").attr("text-anchor","middle").attr("class","packetTitle")},"draw"),U=l((t,e,o,{rowHeight:n,paddingX:i,paddingY:a,bitWidth:r,bitsPerRow:c,showBits:f})=>{const d=t.append("g"),p=o*(n+a)+a;for(const s of e){const k=s.start%c*r+1,g=(s.end-s.start+1)*r-i;if(d.append("rect").attr("x",k).attr("y",p).attr("width",g).attr("height",n).attr("class","packetBlock"),d.append("text").attr("x",k+g/2).attr("y",p+n/2).attr("class","packetLabel").attr("dominant-baseline","middle").attr("text-anchor","middle").text(s.label),!f)continue;const b=s.end===s.start,h=p-2;d.append("text").attr("x",k+(b?g/2:0)).attr("y",h).attr("class","packetByte start").attr("dominant-baseline","auto").attr("text-anchor",b?"middle":"start").text(s.start),b||d.append("text").attr("x",k+g).attr("y",h).attr("class","packetByte end").attr("dominant-baseline","auto").attr("text-anchor","end").text(s.end)}},"drawWord"),X={draw:R},j={byteFontSize:"10px",startByteColor:"black",endByteColor:"black",labelColor:"black",labelFontSize:"12px",titleColor:"black",titleFontSize:"14px",blockStrokeColor:"black",blockStrokeWidth:"1",blockFillColor:"#efefef"},J=l(({packet:t}={})=>{const e=v(j,t);return`
	.packetByte {
		font-size: ${e.byteFontSize};
	}
	.packetByte.start {
		fill: ${e.startByteColor};
	}
	.packetByte.end {
		fill: ${e.endByteColor};
	}
	.packetLabel {
		fill: ${e.labelColor};
		font-size: ${e.labelFontSize};
	}
	.packetTitle {
		fill: ${e.titleColor};
		font-size: ${e.titleFontSize};
	}
	.packetBlock {
		stroke: ${e.blockStrokeColor};
		stroke-width: ${e.blockStrokeWidth};
		fill: ${e.blockFillColor};
	}
	`},"styles"),lt={parser:q,db:u,renderer:X,styles:J};export{lt as diagram};

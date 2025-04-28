import{_ as c,g as ut,s as dt,t as ft,q as ht,a as kt,b as mt,c as ce,d as ge,ay as yt,az as gt,aA as pt,e as vt,R as xt,aB as Tt,aC as X,l as we,aD as bt,aE as qe,aF as Ge,aG as wt,aH as _t,aI as Dt,aJ as Ct,aK as Et,aL as St,aM as Mt,aN as He,aO as Xe,aP as je,aQ as Ue,aR as Ze,aS as It,k as At,j as Ft,z as Lt,u as Yt}from"./mermaid-vendor-BLW4crkP.js";import{g as Ae}from"./react-vendor-DEwriMA6.js";import"./feature-graph-Brx3fnTl.js";import"./graph-vendor-B-X5JegA.js";import"./ui-vendor-CeCm8EER.js";import"./utils-vendor-BysuhMZA.js";var pe={exports:{}},Wt=pe.exports,Qe;function Ot(){return Qe||(Qe=1,function(e,s){(function(n,a){e.exports=a()})(Wt,function(){var n="day";return function(a,i,k){var f=function(S){return S.add(4-S.isoWeekday(),n)},_=i.prototype;_.isoWeekYear=function(){return f(this).year()},_.isoWeek=function(S){if(!this.$utils().u(S))return this.add(7*(S-this.isoWeek()),n);var g,M,P,V,B=f(this),E=(g=this.isoWeekYear(),M=this.$u,P=(M?k.utc:k)().year(g).startOf("year"),V=4-P.isoWeekday(),P.isoWeekday()>4&&(V+=7),P.add(V,n));return B.diff(E,"week")+1},_.isoWeekday=function(S){return this.$utils().u(S)?this.day()||7:this.day(this.day()%7?S:S-7)};var Y=_.startOf;_.startOf=function(S,g){var M=this.$utils(),P=!!M.u(g)||g;return M.p(S)==="isoweek"?P?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):Y.bind(this)(S,g)}}})}(pe)),pe.exports}var Pt=Ot();const Vt=Ae(Pt);var ve={exports:{}},zt=ve.exports,$e;function Rt(){return $e||($e=1,function(e,s){(function(n,a){e.exports=a()})(zt,function(){var n={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},a=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,i=/\d/,k=/\d\d/,f=/\d\d?/,_=/\d*[^-_:/,()\s\d]+/,Y={},S=function(p){return(p=+p)+(p>68?1900:2e3)},g=function(p){return function(C){this[p]=+C}},M=[/[+-]\d\d:?(\d\d)?|Z/,function(p){(this.zone||(this.zone={})).offset=function(C){if(!C||C==="Z")return 0;var F=C.match(/([+-]|\d\d)/g),L=60*F[1]+(+F[2]||0);return L===0?0:F[0]==="+"?-L:L}(p)}],P=function(p){var C=Y[p];return C&&(C.indexOf?C:C.s.concat(C.f))},V=function(p,C){var F,L=Y.meridiem;if(L){for(var G=1;G<=24;G+=1)if(p.indexOf(L(G,0,C))>-1){F=G>12;break}}else F=p===(C?"pm":"PM");return F},B={A:[_,function(p){this.afternoon=V(p,!1)}],a:[_,function(p){this.afternoon=V(p,!0)}],Q:[i,function(p){this.month=3*(p-1)+1}],S:[i,function(p){this.milliseconds=100*+p}],SS:[k,function(p){this.milliseconds=10*+p}],SSS:[/\d{3}/,function(p){this.milliseconds=+p}],s:[f,g("seconds")],ss:[f,g("seconds")],m:[f,g("minutes")],mm:[f,g("minutes")],H:[f,g("hours")],h:[f,g("hours")],HH:[f,g("hours")],hh:[f,g("hours")],D:[f,g("day")],DD:[k,g("day")],Do:[_,function(p){var C=Y.ordinal,F=p.match(/\d+/);if(this.day=F[0],C)for(var L=1;L<=31;L+=1)C(L).replace(/\[|\]/g,"")===p&&(this.day=L)}],w:[f,g("week")],ww:[k,g("week")],M:[f,g("month")],MM:[k,g("month")],MMM:[_,function(p){var C=P("months"),F=(P("monthsShort")||C.map(function(L){return L.slice(0,3)})).indexOf(p)+1;if(F<1)throw new Error;this.month=F%12||F}],MMMM:[_,function(p){var C=P("months").indexOf(p)+1;if(C<1)throw new Error;this.month=C%12||C}],Y:[/[+-]?\d+/,g("year")],YY:[k,function(p){this.year=S(p)}],YYYY:[/\d{4}/,g("year")],Z:M,ZZ:M};function E(p){var C,F;C=p,F=Y&&Y.formats;for(var L=(p=C.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,function(T,b,m){var w=m&&m.toUpperCase();return b||F[m]||n[m]||F[w].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,function(o,l,h){return l||h.slice(1)})})).match(a),G=L.length,H=0;H<G;H+=1){var Q=L[H],j=B[Q],y=j&&j[0],x=j&&j[1];L[H]=x?{regex:y,parser:x}:Q.replace(/^\[|\]$/g,"")}return function(T){for(var b={},m=0,w=0;m<G;m+=1){var o=L[m];if(typeof o=="string")w+=o.length;else{var l=o.regex,h=o.parser,d=T.slice(w),v=l.exec(d)[0];h.call(b,v),T=T.replace(v,"")}}return function(r){var u=r.afternoon;if(u!==void 0){var t=r.hours;u?t<12&&(r.hours+=12):t===12&&(r.hours=0),delete r.afternoon}}(b),b}}return function(p,C,F){F.p.customParseFormat=!0,p&&p.parseTwoDigitYear&&(S=p.parseTwoDigitYear);var L=C.prototype,G=L.parse;L.parse=function(H){var Q=H.date,j=H.utc,y=H.args;this.$u=j;var x=y[1];if(typeof x=="string"){var T=y[2]===!0,b=y[3]===!0,m=T||b,w=y[2];b&&(w=y[2]),Y=this.$locale(),!T&&w&&(Y=F.Ls[w]),this.$d=function(d,v,r,u){try{if(["x","X"].indexOf(v)>-1)return new Date((v==="X"?1e3:1)*d);var t=E(v)(d),I=t.year,D=t.month,A=t.day,N=t.hours,W=t.minutes,O=t.seconds,$=t.milliseconds,ae=t.zone,ie=t.week,de=new Date,fe=A||(I||D?1:de.getDate()),oe=I||de.getFullYear(),z=0;I&&!D||(z=D>0?D-1:de.getMonth());var Z,q=N||0,se=W||0,K=O||0,re=$||0;return ae?new Date(Date.UTC(oe,z,fe,q,se,K,re+60*ae.offset*1e3)):r?new Date(Date.UTC(oe,z,fe,q,se,K,re)):(Z=new Date(oe,z,fe,q,se,K,re),ie&&(Z=u(Z).week(ie).toDate()),Z)}catch{return new Date("")}}(Q,x,j,F),this.init(),w&&w!==!0&&(this.$L=this.locale(w).$L),m&&Q!=this.format(x)&&(this.$d=new Date("")),Y={}}else if(x instanceof Array)for(var o=x.length,l=1;l<=o;l+=1){y[1]=x[l-1];var h=F.apply(this,y);if(h.isValid()){this.$d=h.$d,this.$L=h.$L,this.init();break}l===o&&(this.$d=new Date(""))}else G.call(this,H)}}})}(ve)),ve.exports}var Nt=Rt();const Bt=Ae(Nt);var xe={exports:{}},qt=xe.exports,Ke;function Gt(){return Ke||(Ke=1,function(e,s){(function(n,a){e.exports=a()})(qt,function(){return function(n,a){var i=a.prototype,k=i.format;i.format=function(f){var _=this,Y=this.$locale();if(!this.isValid())return k.bind(this)(f);var S=this.$utils(),g=(f||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,function(M){switch(M){case"Q":return Math.ceil((_.$M+1)/3);case"Do":return Y.ordinal(_.$D);case"gggg":return _.weekYear();case"GGGG":return _.isoWeekYear();case"wo":return Y.ordinal(_.week(),"W");case"w":case"ww":return S.s(_.week(),M==="w"?1:2,"0");case"W":case"WW":return S.s(_.isoWeek(),M==="W"?1:2,"0");case"k":case"kk":return S.s(String(_.$H===0?24:_.$H),M==="k"?1:2,"0");case"X":return Math.floor(_.$d.getTime()/1e3);case"x":return _.$d.getTime();case"z":return"["+_.offsetName()+"]";case"zzz":return"["+_.offsetName("long")+"]";default:return M}});return k.bind(this)(g)}}})}(xe)),xe.exports}var Ht=Gt();const Xt=Ae(Ht);var Ee=function(){var e=c(function(w,o,l,h){for(l=l||{},h=w.length;h--;l[w[h]]=o);return l},"o"),s=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],n=[1,26],a=[1,27],i=[1,28],k=[1,29],f=[1,30],_=[1,31],Y=[1,32],S=[1,33],g=[1,34],M=[1,9],P=[1,10],V=[1,11],B=[1,12],E=[1,13],p=[1,14],C=[1,15],F=[1,16],L=[1,19],G=[1,20],H=[1,21],Q=[1,22],j=[1,23],y=[1,25],x=[1,35],T={trace:c(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:c(function(o,l,h,d,v,r,u){var t=r.length-1;switch(v){case 1:return r[t-1];case 2:this.$=[];break;case 3:r[t-1].push(r[t]),this.$=r[t-1];break;case 4:case 5:this.$=r[t];break;case 6:case 7:this.$=[];break;case 8:d.setWeekday("monday");break;case 9:d.setWeekday("tuesday");break;case 10:d.setWeekday("wednesday");break;case 11:d.setWeekday("thursday");break;case 12:d.setWeekday("friday");break;case 13:d.setWeekday("saturday");break;case 14:d.setWeekday("sunday");break;case 15:d.setWeekend("friday");break;case 16:d.setWeekend("saturday");break;case 17:d.setDateFormat(r[t].substr(11)),this.$=r[t].substr(11);break;case 18:d.enableInclusiveEndDates(),this.$=r[t].substr(18);break;case 19:d.TopAxis(),this.$=r[t].substr(8);break;case 20:d.setAxisFormat(r[t].substr(11)),this.$=r[t].substr(11);break;case 21:d.setTickInterval(r[t].substr(13)),this.$=r[t].substr(13);break;case 22:d.setExcludes(r[t].substr(9)),this.$=r[t].substr(9);break;case 23:d.setIncludes(r[t].substr(9)),this.$=r[t].substr(9);break;case 24:d.setTodayMarker(r[t].substr(12)),this.$=r[t].substr(12);break;case 27:d.setDiagramTitle(r[t].substr(6)),this.$=r[t].substr(6);break;case 28:this.$=r[t].trim(),d.setAccTitle(this.$);break;case 29:case 30:this.$=r[t].trim(),d.setAccDescription(this.$);break;case 31:d.addSection(r[t].substr(8)),this.$=r[t].substr(8);break;case 33:d.addTask(r[t-1],r[t]),this.$="task";break;case 34:this.$=r[t-1],d.setClickEvent(r[t-1],r[t],null);break;case 35:this.$=r[t-2],d.setClickEvent(r[t-2],r[t-1],r[t]);break;case 36:this.$=r[t-2],d.setClickEvent(r[t-2],r[t-1],null),d.setLink(r[t-2],r[t]);break;case 37:this.$=r[t-3],d.setClickEvent(r[t-3],r[t-2],r[t-1]),d.setLink(r[t-3],r[t]);break;case 38:this.$=r[t-2],d.setClickEvent(r[t-2],r[t],null),d.setLink(r[t-2],r[t-1]);break;case 39:this.$=r[t-3],d.setClickEvent(r[t-3],r[t-1],r[t]),d.setLink(r[t-3],r[t-2]);break;case 40:this.$=r[t-1],d.setLink(r[t-1],r[t]);break;case 41:case 47:this.$=r[t-1]+" "+r[t];break;case 42:case 43:case 45:this.$=r[t-2]+" "+r[t-1]+" "+r[t];break;case 44:case 46:this.$=r[t-3]+" "+r[t-2]+" "+r[t-1]+" "+r[t];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},e(s,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:n,13:a,14:i,15:k,16:f,17:_,18:Y,19:18,20:S,21:g,22:M,23:P,24:V,25:B,26:E,27:p,28:C,29:F,30:L,31:G,33:H,35:Q,36:j,37:24,38:y,40:x},e(s,[2,7],{1:[2,1]}),e(s,[2,3]),{9:36,11:17,12:n,13:a,14:i,15:k,16:f,17:_,18:Y,19:18,20:S,21:g,22:M,23:P,24:V,25:B,26:E,27:p,28:C,29:F,30:L,31:G,33:H,35:Q,36:j,37:24,38:y,40:x},e(s,[2,5]),e(s,[2,6]),e(s,[2,17]),e(s,[2,18]),e(s,[2,19]),e(s,[2,20]),e(s,[2,21]),e(s,[2,22]),e(s,[2,23]),e(s,[2,24]),e(s,[2,25]),e(s,[2,26]),e(s,[2,27]),{32:[1,37]},{34:[1,38]},e(s,[2,30]),e(s,[2,31]),e(s,[2,32]),{39:[1,39]},e(s,[2,8]),e(s,[2,9]),e(s,[2,10]),e(s,[2,11]),e(s,[2,12]),e(s,[2,13]),e(s,[2,14]),e(s,[2,15]),e(s,[2,16]),{41:[1,40],43:[1,41]},e(s,[2,4]),e(s,[2,28]),e(s,[2,29]),e(s,[2,33]),e(s,[2,34],{42:[1,42],43:[1,43]}),e(s,[2,40],{41:[1,44]}),e(s,[2,35],{43:[1,45]}),e(s,[2,36]),e(s,[2,38],{42:[1,46]}),e(s,[2,37]),e(s,[2,39])],defaultActions:{},parseError:c(function(o,l){if(l.recoverable)this.trace(o);else{var h=new Error(o);throw h.hash=l,h}},"parseError"),parse:c(function(o){var l=this,h=[0],d=[],v=[null],r=[],u=this.table,t="",I=0,D=0,A=2,N=1,W=r.slice.call(arguments,1),O=Object.create(this.lexer),$={yy:{}};for(var ae in this.yy)Object.prototype.hasOwnProperty.call(this.yy,ae)&&($.yy[ae]=this.yy[ae]);O.setInput(o,$.yy),$.yy.lexer=O,$.yy.parser=this,typeof O.yylloc>"u"&&(O.yylloc={});var ie=O.yylloc;r.push(ie);var de=O.options&&O.options.ranges;typeof $.yy.parseError=="function"?this.parseError=$.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function fe(U){h.length=h.length-2*U,v.length=v.length-U,r.length=r.length-U}c(fe,"popStack");function oe(){var U;return U=d.pop()||O.lex()||N,typeof U!="number"&&(U instanceof Array&&(d=U,U=d.pop()),U=l.symbols_[U]||U),U}c(oe,"lex");for(var z,Z,q,se,K={},re,J,Be,ye;;){if(Z=h[h.length-1],this.defaultActions[Z]?q=this.defaultActions[Z]:((z===null||typeof z>"u")&&(z=oe()),q=u[Z]&&u[Z][z]),typeof q>"u"||!q.length||!q[0]){var Ce="";ye=[];for(re in u[Z])this.terminals_[re]&&re>A&&ye.push("'"+this.terminals_[re]+"'");O.showPosition?Ce="Parse error on line "+(I+1)+`:
`+O.showPosition()+`
Expecting `+ye.join(", ")+", got '"+(this.terminals_[z]||z)+"'":Ce="Parse error on line "+(I+1)+": Unexpected "+(z==N?"end of input":"'"+(this.terminals_[z]||z)+"'"),this.parseError(Ce,{text:O.match,token:this.terminals_[z]||z,line:O.yylineno,loc:ie,expected:ye})}if(q[0]instanceof Array&&q.length>1)throw new Error("Parse Error: multiple actions possible at state: "+Z+", token: "+z);switch(q[0]){case 1:h.push(z),v.push(O.yytext),r.push(O.yylloc),h.push(q[1]),z=null,D=O.yyleng,t=O.yytext,I=O.yylineno,ie=O.yylloc;break;case 2:if(J=this.productions_[q[1]][1],K.$=v[v.length-J],K._$={first_line:r[r.length-(J||1)].first_line,last_line:r[r.length-1].last_line,first_column:r[r.length-(J||1)].first_column,last_column:r[r.length-1].last_column},de&&(K._$.range=[r[r.length-(J||1)].range[0],r[r.length-1].range[1]]),se=this.performAction.apply(K,[t,D,I,$.yy,q[1],v,r].concat(W)),typeof se<"u")return se;J&&(h=h.slice(0,-1*J*2),v=v.slice(0,-1*J),r=r.slice(0,-1*J)),h.push(this.productions_[q[1]][0]),v.push(K.$),r.push(K._$),Be=u[h[h.length-2]][h[h.length-1]],h.push(Be);break;case 3:return!0}}return!0},"parse")},b=function(){var w={EOF:1,parseError:c(function(l,h){if(this.yy.parser)this.yy.parser.parseError(l,h);else throw new Error(l)},"parseError"),setInput:c(function(o,l){return this.yy=l||this.yy||{},this._input=o,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:c(function(){var o=this._input[0];this.yytext+=o,this.yyleng++,this.offset++,this.match+=o,this.matched+=o;var l=o.match(/(?:\r\n?|\n).*/g);return l?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),o},"input"),unput:c(function(o){var l=o.length,h=o.split(/(?:\r\n?|\n)/g);this._input=o+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-l),this.offset-=l;var d=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),h.length-1&&(this.yylineno-=h.length-1);var v=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:h?(h.length===d.length?this.yylloc.first_column:0)+d[d.length-h.length].length-h[0].length:this.yylloc.first_column-l},this.options.ranges&&(this.yylloc.range=[v[0],v[0]+this.yyleng-l]),this.yyleng=this.yytext.length,this},"unput"),more:c(function(){return this._more=!0,this},"more"),reject:c(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:c(function(o){this.unput(this.match.slice(o))},"less"),pastInput:c(function(){var o=this.matched.substr(0,this.matched.length-this.match.length);return(o.length>20?"...":"")+o.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:c(function(){var o=this.match;return o.length<20&&(o+=this._input.substr(0,20-o.length)),(o.substr(0,20)+(o.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:c(function(){var o=this.pastInput(),l=new Array(o.length+1).join("-");return o+this.upcomingInput()+`
`+l+"^"},"showPosition"),test_match:c(function(o,l){var h,d,v;if(this.options.backtrack_lexer&&(v={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(v.yylloc.range=this.yylloc.range.slice(0))),d=o[0].match(/(?:\r\n?|\n).*/g),d&&(this.yylineno+=d.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:d?d[d.length-1].length-d[d.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+o[0].length},this.yytext+=o[0],this.match+=o[0],this.matches=o,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(o[0].length),this.matched+=o[0],h=this.performAction.call(this,this.yy,this,l,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),h)return h;if(this._backtrack){for(var r in v)this[r]=v[r];return!1}return!1},"test_match"),next:c(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var o,l,h,d;this._more||(this.yytext="",this.match="");for(var v=this._currentRules(),r=0;r<v.length;r++)if(h=this._input.match(this.rules[v[r]]),h&&(!l||h[0].length>l[0].length)){if(l=h,d=r,this.options.backtrack_lexer){if(o=this.test_match(h,v[r]),o!==!1)return o;if(this._backtrack){l=!1;continue}else return!1}else if(!this.options.flex)break}return l?(o=this.test_match(l,v[d]),o!==!1?o:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:c(function(){var l=this.next();return l||this.lex()},"lex"),begin:c(function(l){this.conditionStack.push(l)},"begin"),popState:c(function(){var l=this.conditionStack.length-1;return l>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:c(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:c(function(l){return l=this.conditionStack.length-1-Math.abs(l||0),l>=0?this.conditionStack[l]:"INITIAL"},"topState"),pushState:c(function(l){this.begin(l)},"pushState"),stateStackSize:c(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:c(function(l,h,d,v){switch(d){case 0:return this.begin("open_directive"),"open_directive";case 1:return this.begin("acc_title"),31;case 2:return this.popState(),"acc_title_value";case 3:return this.begin("acc_descr"),33;case 4:return this.popState(),"acc_descr_value";case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}};return w}();T.lexer=b;function m(){this.yy={}}return c(m,"Parser"),m.prototype=T,T.Parser=m,new m}();Ee.parser=Ee;var jt=Ee;X.extend(Vt);X.extend(Bt);X.extend(Xt);var Je={friday:5,saturday:6},ee="",Fe="",Le=void 0,Ye="",he=[],ke=[],We=new Map,Oe=[],_e=[],ue="",Pe="",rt=["active","done","crit","milestone"],Ve=[],me=!1,ze=!1,Re="sunday",De="saturday",Se=0,Ut=c(function(){Oe=[],_e=[],ue="",Ve=[],Te=0,Ie=void 0,be=void 0,R=[],ee="",Fe="",Pe="",Le=void 0,Ye="",he=[],ke=[],me=!1,ze=!1,Se=0,We=new Map,Lt(),Re="sunday",De="saturday"},"clear"),Zt=c(function(e){Fe=e},"setAxisFormat"),Qt=c(function(){return Fe},"getAxisFormat"),$t=c(function(e){Le=e},"setTickInterval"),Kt=c(function(){return Le},"getTickInterval"),Jt=c(function(e){Ye=e},"setTodayMarker"),er=c(function(){return Ye},"getTodayMarker"),tr=c(function(e){ee=e},"setDateFormat"),rr=c(function(){me=!0},"enableInclusiveEndDates"),sr=c(function(){return me},"endDatesAreInclusive"),nr=c(function(){ze=!0},"enableTopAxis"),ar=c(function(){return ze},"topAxisEnabled"),ir=c(function(e){Pe=e},"setDisplayMode"),or=c(function(){return Pe},"getDisplayMode"),cr=c(function(){return ee},"getDateFormat"),lr=c(function(e){he=e.toLowerCase().split(/[\s,]+/)},"setIncludes"),ur=c(function(){return he},"getIncludes"),dr=c(function(e){ke=e.toLowerCase().split(/[\s,]+/)},"setExcludes"),fr=c(function(){return ke},"getExcludes"),hr=c(function(){return We},"getLinks"),kr=c(function(e){ue=e,Oe.push(e)},"addSection"),mr=c(function(){return Oe},"getSections"),yr=c(function(){let e=et();const s=10;let n=0;for(;!e&&n<s;)e=et(),n++;return _e=R,_e},"getTasks"),st=c(function(e,s,n,a){return a.includes(e.format(s.trim()))?!1:n.includes("weekends")&&(e.isoWeekday()===Je[De]||e.isoWeekday()===Je[De]+1)||n.includes(e.format("dddd").toLowerCase())?!0:n.includes(e.format(s.trim()))},"isInvalidDate"),gr=c(function(e){Re=e},"setWeekday"),pr=c(function(){return Re},"getWeekday"),vr=c(function(e){De=e},"setWeekend"),nt=c(function(e,s,n,a){if(!n.length||e.manualEndTime)return;let i;e.startTime instanceof Date?i=X(e.startTime):i=X(e.startTime,s,!0),i=i.add(1,"d");let k;e.endTime instanceof Date?k=X(e.endTime):k=X(e.endTime,s,!0);const[f,_]=xr(i,k,s,n,a);e.endTime=f.toDate(),e.renderEndTime=_},"checkTaskDates"),xr=c(function(e,s,n,a,i){let k=!1,f=null;for(;e<=s;)k||(f=s.toDate()),k=st(e,n,a,i),k&&(s=s.add(1,"d")),e=e.add(1,"d");return[s,f]},"fixTaskDates"),Me=c(function(e,s,n){n=n.trim();const i=/^after\s+(?<ids>[\d\w- ]+)/.exec(n);if(i!==null){let f=null;for(const Y of i.groups.ids.split(" ")){let S=ne(Y);S!==void 0&&(!f||S.endTime>f.endTime)&&(f=S)}if(f)return f.endTime;const _=new Date;return _.setHours(0,0,0,0),_}let k=X(n,s.trim(),!0);if(k.isValid())return k.toDate();{we.debug("Invalid date:"+n),we.debug("With date format:"+s.trim());const f=new Date(n);if(f===void 0||isNaN(f.getTime())||f.getFullYear()<-1e4||f.getFullYear()>1e4)throw new Error("Invalid date:"+n);return f}},"getStartDate"),at=c(function(e){const s=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(e.trim());return s!==null?[Number.parseFloat(s[1]),s[2]]:[NaN,"ms"]},"parseDuration"),it=c(function(e,s,n,a=!1){n=n.trim();const k=/^until\s+(?<ids>[\d\w- ]+)/.exec(n);if(k!==null){let g=null;for(const P of k.groups.ids.split(" ")){let V=ne(P);V!==void 0&&(!g||V.startTime<g.startTime)&&(g=V)}if(g)return g.startTime;const M=new Date;return M.setHours(0,0,0,0),M}let f=X(n,s.trim(),!0);if(f.isValid())return a&&(f=f.add(1,"d")),f.toDate();let _=X(e);const[Y,S]=at(n);if(!Number.isNaN(Y)){const g=_.add(Y,S);g.isValid()&&(_=g)}return _.toDate()},"getEndDate"),Te=0,le=c(function(e){return e===void 0?(Te=Te+1,"task"+Te):e},"parseId"),Tr=c(function(e,s){let n;s.substr(0,1)===":"?n=s.substr(1,s.length):n=s;const a=n.split(","),i={};Ne(a,i,rt);for(let f=0;f<a.length;f++)a[f]=a[f].trim();let k="";switch(a.length){case 1:i.id=le(),i.startTime=e.endTime,k=a[0];break;case 2:i.id=le(),i.startTime=Me(void 0,ee,a[0]),k=a[1];break;case 3:i.id=le(a[0]),i.startTime=Me(void 0,ee,a[1]),k=a[2];break}return k&&(i.endTime=it(i.startTime,ee,k,me),i.manualEndTime=X(k,"YYYY-MM-DD",!0).isValid(),nt(i,ee,ke,he)),i},"compileData"),br=c(function(e,s){let n;s.substr(0,1)===":"?n=s.substr(1,s.length):n=s;const a=n.split(","),i={};Ne(a,i,rt);for(let k=0;k<a.length;k++)a[k]=a[k].trim();switch(a.length){case 1:i.id=le(),i.startTime={type:"prevTaskEnd",id:e},i.endTime={data:a[0]};break;case 2:i.id=le(),i.startTime={type:"getStartDate",startData:a[0]},i.endTime={data:a[1]};break;case 3:i.id=le(a[0]),i.startTime={type:"getStartDate",startData:a[1]},i.endTime={data:a[2]};break}return i},"parseData"),Ie,be,R=[],ot={},wr=c(function(e,s){const n={section:ue,type:ue,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:s},task:e,classes:[]},a=br(be,s);n.raw.startTime=a.startTime,n.raw.endTime=a.endTime,n.id=a.id,n.prevTaskId=be,n.active=a.active,n.done=a.done,n.crit=a.crit,n.milestone=a.milestone,n.order=Se,Se++;const i=R.push(n);be=n.id,ot[n.id]=i-1},"addTask"),ne=c(function(e){const s=ot[e];return R[s]},"findTaskById"),_r=c(function(e,s){const n={section:ue,type:ue,description:e,task:e,classes:[]},a=Tr(Ie,s);n.startTime=a.startTime,n.endTime=a.endTime,n.id=a.id,n.active=a.active,n.done=a.done,n.crit=a.crit,n.milestone=a.milestone,Ie=n,_e.push(n)},"addTaskOrg"),et=c(function(){const e=c(function(n){const a=R[n];let i="";switch(R[n].raw.startTime.type){case"prevTaskEnd":{const k=ne(a.prevTaskId);a.startTime=k.endTime;break}case"getStartDate":i=Me(void 0,ee,R[n].raw.startTime.startData),i&&(R[n].startTime=i);break}return R[n].startTime&&(R[n].endTime=it(R[n].startTime,ee,R[n].raw.endTime.data,me),R[n].endTime&&(R[n].processed=!0,R[n].manualEndTime=X(R[n].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),nt(R[n],ee,ke,he))),R[n].processed},"compileTask");let s=!0;for(const[n,a]of R.entries())e(n),s=s&&a.processed;return s},"compileTasks"),Dr=c(function(e,s){let n=s;ce().securityLevel!=="loose"&&(n=Ft.sanitizeUrl(s)),e.split(",").forEach(function(a){ne(a)!==void 0&&(lt(a,()=>{window.open(n,"_self")}),We.set(a,n))}),ct(e,"clickable")},"setLink"),ct=c(function(e,s){e.split(",").forEach(function(n){let a=ne(n);a!==void 0&&a.classes.push(s)})},"setClass"),Cr=c(function(e,s,n){if(ce().securityLevel!=="loose"||s===void 0)return;let a=[];if(typeof n=="string"){a=n.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let k=0;k<a.length;k++){let f=a[k].trim();f.startsWith('"')&&f.endsWith('"')&&(f=f.substr(1,f.length-2)),a[k]=f}}a.length===0&&a.push(e),ne(e)!==void 0&&lt(e,()=>{Yt.runFunc(s,...a)})},"setClickFun"),lt=c(function(e,s){Ve.push(function(){const n=document.querySelector(`[id="${e}"]`);n!==null&&n.addEventListener("click",function(){s()})},function(){const n=document.querySelector(`[id="${e}-text"]`);n!==null&&n.addEventListener("click",function(){s()})})},"pushFun"),Er=c(function(e,s,n){e.split(",").forEach(function(a){Cr(a,s,n)}),ct(e,"clickable")},"setClickEvent"),Sr=c(function(e){Ve.forEach(function(s){s(e)})},"bindFunctions"),Mr={getConfig:c(()=>ce().gantt,"getConfig"),clear:Ut,setDateFormat:tr,getDateFormat:cr,enableInclusiveEndDates:rr,endDatesAreInclusive:sr,enableTopAxis:nr,topAxisEnabled:ar,setAxisFormat:Zt,getAxisFormat:Qt,setTickInterval:$t,getTickInterval:Kt,setTodayMarker:Jt,getTodayMarker:er,setAccTitle:mt,getAccTitle:kt,setDiagramTitle:ht,getDiagramTitle:ft,setDisplayMode:ir,getDisplayMode:or,setAccDescription:dt,getAccDescription:ut,addSection:kr,getSections:mr,getTasks:yr,addTask:wr,findTaskById:ne,addTaskOrg:_r,setIncludes:lr,getIncludes:ur,setExcludes:dr,getExcludes:fr,setClickEvent:Er,setLink:Dr,getLinks:hr,bindFunctions:Sr,parseDuration:at,isInvalidDate:st,setWeekday:gr,getWeekday:pr,setWeekend:vr};function Ne(e,s,n){let a=!0;for(;a;)a=!1,n.forEach(function(i){const k="^\\s*"+i+"\\s*$",f=new RegExp(k);e[0].match(f)&&(s[i]=!0,e.shift(1),a=!0)})}c(Ne,"getTaskTags");var Ir=c(function(){we.debug("Something is calling, setConf, remove the call")},"setConf"),tt={monday:Mt,tuesday:St,wednesday:Et,thursday:Ct,friday:Dt,saturday:_t,sunday:wt},Ar=c((e,s)=>{let n=[...e].map(()=>-1/0),a=[...e].sort((k,f)=>k.startTime-f.startTime||k.order-f.order),i=0;for(const k of a)for(let f=0;f<n.length;f++)if(k.startTime>=n[f]){n[f]=k.endTime,k.order=f+s,f>i&&(i=f);break}return i},"getMaxIntersections"),te,Fr=c(function(e,s,n,a){const i=ce().gantt,k=ce().securityLevel;let f;k==="sandbox"&&(f=ge("#i"+s));const _=k==="sandbox"?ge(f.nodes()[0].contentDocument.body):ge("body"),Y=k==="sandbox"?f.nodes()[0].contentDocument:document,S=Y.getElementById(s);te=S.parentElement.offsetWidth,te===void 0&&(te=1200),i.useWidth!==void 0&&(te=i.useWidth);const g=a.db.getTasks();let M=[];for(const y of g)M.push(y.type);M=j(M);const P={};let V=2*i.topPadding;if(a.db.getDisplayMode()==="compact"||i.displayMode==="compact"){const y={};for(const T of g)y[T.section]===void 0?y[T.section]=[T]:y[T.section].push(T);let x=0;for(const T of Object.keys(y)){const b=Ar(y[T],x)+1;x+=b,V+=b*(i.barHeight+i.barGap),P[T]=b}}else{V+=g.length*(i.barHeight+i.barGap);for(const y of M)P[y]=g.filter(x=>x.type===y).length}S.setAttribute("viewBox","0 0 "+te+" "+V);const B=_.select(`[id="${s}"]`),E=yt().domain([gt(g,function(y){return y.startTime}),pt(g,function(y){return y.endTime})]).rangeRound([0,te-i.leftPadding-i.rightPadding]);function p(y,x){const T=y.startTime,b=x.startTime;let m=0;return T>b?m=1:T<b&&(m=-1),m}c(p,"taskCompare"),g.sort(p),C(g,te,V),vt(B,V,te,i.useMaxWidth),B.append("text").text(a.db.getDiagramTitle()).attr("x",te/2).attr("y",i.titleTopMargin).attr("class","titleText");function C(y,x,T){const b=i.barHeight,m=b+i.barGap,w=i.topPadding,o=i.leftPadding,l=xt().domain([0,M.length]).range(["#00B9FA","#F95002"]).interpolate(Tt);L(m,w,o,x,T,y,a.db.getExcludes(),a.db.getIncludes()),G(o,w,x,T),F(y,m,w,o,b,l,x),H(m,w),Q(o,w,x,T)}c(C,"makeGantt");function F(y,x,T,b,m,w,o){const h=[...new Set(y.map(u=>u.order))].map(u=>y.find(t=>t.order===u));B.append("g").selectAll("rect").data(h).enter().append("rect").attr("x",0).attr("y",function(u,t){return t=u.order,t*x+T-2}).attr("width",function(){return o-i.rightPadding/2}).attr("height",x).attr("class",function(u){for(const[t,I]of M.entries())if(u.type===I)return"section section"+t%i.numberSectionStyles;return"section section0"});const d=B.append("g").selectAll("rect").data(y).enter(),v=a.db.getLinks();if(d.append("rect").attr("id",function(u){return u.id}).attr("rx",3).attr("ry",3).attr("x",function(u){return u.milestone?E(u.startTime)+b+.5*(E(u.endTime)-E(u.startTime))-.5*m:E(u.startTime)+b}).attr("y",function(u,t){return t=u.order,t*x+T}).attr("width",function(u){return u.milestone?m:E(u.renderEndTime||u.endTime)-E(u.startTime)}).attr("height",m).attr("transform-origin",function(u,t){return t=u.order,(E(u.startTime)+b+.5*(E(u.endTime)-E(u.startTime))).toString()+"px "+(t*x+T+.5*m).toString()+"px"}).attr("class",function(u){const t="task";let I="";u.classes.length>0&&(I=u.classes.join(" "));let D=0;for(const[N,W]of M.entries())u.type===W&&(D=N%i.numberSectionStyles);let A="";return u.active?u.crit?A+=" activeCrit":A=" active":u.done?u.crit?A=" doneCrit":A=" done":u.crit&&(A+=" crit"),A.length===0&&(A=" task"),u.milestone&&(A=" milestone "+A),A+=D,A+=" "+I,t+A}),d.append("text").attr("id",function(u){return u.id+"-text"}).text(function(u){return u.task}).attr("font-size",i.fontSize).attr("x",function(u){let t=E(u.startTime),I=E(u.renderEndTime||u.endTime);u.milestone&&(t+=.5*(E(u.endTime)-E(u.startTime))-.5*m),u.milestone&&(I=t+m);const D=this.getBBox().width;return D>I-t?I+D+1.5*i.leftPadding>o?t+b-5:I+b+5:(I-t)/2+t+b}).attr("y",function(u,t){return t=u.order,t*x+i.barHeight/2+(i.fontSize/2-2)+T}).attr("text-height",m).attr("class",function(u){const t=E(u.startTime);let I=E(u.endTime);u.milestone&&(I=t+m);const D=this.getBBox().width;let A="";u.classes.length>0&&(A=u.classes.join(" "));let N=0;for(const[O,$]of M.entries())u.type===$&&(N=O%i.numberSectionStyles);let W="";return u.active&&(u.crit?W="activeCritText"+N:W="activeText"+N),u.done?u.crit?W=W+" doneCritText"+N:W=W+" doneText"+N:u.crit&&(W=W+" critText"+N),u.milestone&&(W+=" milestoneText"),D>I-t?I+D+1.5*i.leftPadding>o?A+" taskTextOutsideLeft taskTextOutside"+N+" "+W:A+" taskTextOutsideRight taskTextOutside"+N+" "+W+" width-"+D:A+" taskText taskText"+N+" "+W+" width-"+D}),ce().securityLevel==="sandbox"){let u;u=ge("#i"+s);const t=u.nodes()[0].contentDocument;d.filter(function(I){return v.has(I.id)}).each(function(I){var D=t.querySelector("#"+I.id),A=t.querySelector("#"+I.id+"-text");const N=D.parentNode;var W=t.createElement("a");W.setAttribute("xlink:href",v.get(I.id)),W.setAttribute("target","_top"),N.appendChild(W),W.appendChild(D),W.appendChild(A)})}}c(F,"drawRects");function L(y,x,T,b,m,w,o,l){if(o.length===0&&l.length===0)return;let h,d;for(const{startTime:D,endTime:A}of w)(h===void 0||D<h)&&(h=D),(d===void 0||A>d)&&(d=A);if(!h||!d)return;if(X(d).diff(X(h),"year")>5){we.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}const v=a.db.getDateFormat(),r=[];let u=null,t=X(h);for(;t.valueOf()<=d;)a.db.isInvalidDate(t,v,o,l)?u?u.end=t:u={start:t,end:t}:u&&(r.push(u),u=null),t=t.add(1,"d");B.append("g").selectAll("rect").data(r).enter().append("rect").attr("id",function(D){return"exclude-"+D.start.format("YYYY-MM-DD")}).attr("x",function(D){return E(D.start)+T}).attr("y",i.gridLineStartPadding).attr("width",function(D){const A=D.end.add(1,"day");return E(A)-E(D.start)}).attr("height",m-x-i.gridLineStartPadding).attr("transform-origin",function(D,A){return(E(D.start)+T+.5*(E(D.end)-E(D.start))).toString()+"px "+(A*y+.5*m).toString()+"px"}).attr("class","exclude-range")}c(L,"drawExcludeDays");function G(y,x,T,b){let m=bt(E).tickSize(-b+x+i.gridLineStartPadding).tickFormat(qe(a.db.getAxisFormat()||i.axisFormat||"%Y-%m-%d"));const o=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(a.db.getTickInterval()||i.tickInterval);if(o!==null){const l=o[1],h=o[2],d=a.db.getWeekday()||i.weekday;switch(h){case"millisecond":m.ticks(Ze.every(l));break;case"second":m.ticks(Ue.every(l));break;case"minute":m.ticks(je.every(l));break;case"hour":m.ticks(Xe.every(l));break;case"day":m.ticks(He.every(l));break;case"week":m.ticks(tt[d].every(l));break;case"month":m.ticks(Ge.every(l));break}}if(B.append("g").attr("class","grid").attr("transform","translate("+y+", "+(b-50)+")").call(m).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),a.db.topAxisEnabled()||i.topAxis){let l=It(E).tickSize(-b+x+i.gridLineStartPadding).tickFormat(qe(a.db.getAxisFormat()||i.axisFormat||"%Y-%m-%d"));if(o!==null){const h=o[1],d=o[2],v=a.db.getWeekday()||i.weekday;switch(d){case"millisecond":l.ticks(Ze.every(h));break;case"second":l.ticks(Ue.every(h));break;case"minute":l.ticks(je.every(h));break;case"hour":l.ticks(Xe.every(h));break;case"day":l.ticks(He.every(h));break;case"week":l.ticks(tt[v].every(h));break;case"month":l.ticks(Ge.every(h));break}}B.append("g").attr("class","grid").attr("transform","translate("+y+", "+x+")").call(l).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}c(G,"makeGrid");function H(y,x){let T=0;const b=Object.keys(P).map(m=>[m,P[m]]);B.append("g").selectAll("text").data(b).enter().append(function(m){const w=m[0].split(At.lineBreakRegex),o=-(w.length-1)/2,l=Y.createElementNS("http://www.w3.org/2000/svg","text");l.setAttribute("dy",o+"em");for(const[h,d]of w.entries()){const v=Y.createElementNS("http://www.w3.org/2000/svg","tspan");v.setAttribute("alignment-baseline","central"),v.setAttribute("x","10"),h>0&&v.setAttribute("dy","1em"),v.textContent=d,l.appendChild(v)}return l}).attr("x",10).attr("y",function(m,w){if(w>0)for(let o=0;o<w;o++)return T+=b[w-1][1],m[1]*y/2+T*y+x;else return m[1]*y/2+x}).attr("font-size",i.sectionFontSize).attr("class",function(m){for(const[w,o]of M.entries())if(m[0]===o)return"sectionTitle sectionTitle"+w%i.numberSectionStyles;return"sectionTitle"})}c(H,"vertLabels");function Q(y,x,T,b){const m=a.db.getTodayMarker();if(m==="off")return;const w=B.append("g").attr("class","today"),o=new Date,l=w.append("line");l.attr("x1",E(o)+y).attr("x2",E(o)+y).attr("y1",i.titleTopMargin).attr("y2",b-i.titleTopMargin).attr("class","today"),m!==""&&l.attr("style",m.replace(/,/g,";"))}c(Q,"drawToday");function j(y){const x={},T=[];for(let b=0,m=y.length;b<m;++b)Object.prototype.hasOwnProperty.call(x,y[b])||(x[y[b]]=!0,T.push(y[b]));return T}c(j,"checkUnique")},"draw"),Lr={setConf:Ir,draw:Fr},Yr=c(e=>`
  .mermaid-main-font {
        font-family: ${e.fontFamily};
  }

  .exclude-range {
    fill: ${e.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${e.sectionBkgColor};
  }

  .section2 {
    fill: ${e.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${e.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${e.titleColor};
  }

  .sectionTitle1 {
    fill: ${e.titleColor};
  }

  .sectionTitle2 {
    fill: ${e.titleColor};
  }

  .sectionTitle3 {
    fill: ${e.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${e.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${e.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${e.fontFamily};
    fill: ${e.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${e.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${e.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${e.taskTextDarkColor};
    text-anchor: start;
    font-family: ${e.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${e.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${e.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${e.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${e.taskBkgColor};
    stroke: ${e.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${e.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${e.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${e.activeTaskBkgColor};
    stroke: ${e.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${e.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${e.doneTaskBorderColor};
    fill: ${e.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${e.taskTextDarkColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${e.critBorderColor};
    fill: ${e.doneTaskBkgColor};
    stroke-width: 2;
    cursor: pointer;
    shape-rendering: crispEdges;
  }

  .milestone {
    transform: rotate(45deg) scale(0.8,0.8);
  }

  .milestoneText {
    font-style: italic;
  }
  .doneCritText0,
  .doneCritText1,
  .doneCritText2,
  .doneCritText3 {
    fill: ${e.taskTextDarkColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${e.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${e.titleColor||e.textColor};
    font-family: ${e.fontFamily};
  }
`,"getStyles"),Wr=Yr,Br={parser:jt,db:Mr,renderer:Lr,styles:Wr};export{Br as diagram};

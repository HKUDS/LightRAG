import{_ as c,g as ue,s as de,t as fe,q as he,a as ke,b as me,c as ct,d as gt,ay as ye,az as ge,aA as pe,e as ve,R as xe,aB as Te,aC as X,l as wt,aD as be,aE as qt,aF as Gt,aG as we,aH as _e,aI as De,aJ as Ce,aK as Ee,aL as Se,aM as Me,aN as Ht,aO as Xt,aP as jt,aQ as Ut,aR as Zt,aS as Ie,k as Ae,j as Fe,z as Le,u as Ye}from"./mermaid-vendor-C0Jd7kiO.js";import{g as At}from"./react-vendor-By5LtphY.js";import"./feature-graph-CZbDrq4y.js";import"./graph-vendor-DU5LONRU.js";import"./ui-vendor-CyXFCsv0.js";import"./utils-vendor-CoZEqXwV.js";var pt={exports:{}},We=pt.exports,Qt;function Oe(){return Qt||(Qt=1,function(t,s){(function(a,i){t.exports=i()})(We,function(){var a="day";return function(i,n,k){var f=function(S){return S.add(4-S.isoWeekday(),a)},_=n.prototype;_.isoWeekYear=function(){return f(this).year()},_.isoWeek=function(S){if(!this.$utils().u(S))return this.add(7*(S-this.isoWeek()),a);var g,M,P,V,B=f(this),E=(g=this.isoWeekYear(),M=this.$u,P=(M?k.utc:k)().year(g).startOf("year"),V=4-P.isoWeekday(),P.isoWeekday()>4&&(V+=7),P.add(V,a));return B.diff(E,"week")+1},_.isoWeekday=function(S){return this.$utils().u(S)?this.day()||7:this.day(this.day()%7?S:S-7)};var Y=_.startOf;_.startOf=function(S,g){var M=this.$utils(),P=!!M.u(g)||g;return M.p(S)==="isoweek"?P?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):Y.bind(this)(S,g)}}})}(pt)),pt.exports}var Pe=Oe();const Ve=At(Pe);var vt={exports:{}},ze=vt.exports,$t;function Re(){return $t||($t=1,function(t,s){(function(a,i){t.exports=i()})(ze,function(){var a={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},i=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,n=/\d/,k=/\d\d/,f=/\d\d?/,_=/\d*[^-_:/,()\s\d]+/,Y={},S=function(p){return(p=+p)+(p>68?1900:2e3)},g=function(p){return function(C){this[p]=+C}},M=[/[+-]\d\d:?(\d\d)?|Z/,function(p){(this.zone||(this.zone={})).offset=function(C){if(!C||C==="Z")return 0;var F=C.match(/([+-]|\d\d)/g),L=60*F[1]+(+F[2]||0);return L===0?0:F[0]==="+"?-L:L}(p)}],P=function(p){var C=Y[p];return C&&(C.indexOf?C:C.s.concat(C.f))},V=function(p,C){var F,L=Y.meridiem;if(L){for(var G=1;G<=24;G+=1)if(p.indexOf(L(G,0,C))>-1){F=G>12;break}}else F=p===(C?"pm":"PM");return F},B={A:[_,function(p){this.afternoon=V(p,!1)}],a:[_,function(p){this.afternoon=V(p,!0)}],Q:[n,function(p){this.month=3*(p-1)+1}],S:[n,function(p){this.milliseconds=100*+p}],SS:[k,function(p){this.milliseconds=10*+p}],SSS:[/\d{3}/,function(p){this.milliseconds=+p}],s:[f,g("seconds")],ss:[f,g("seconds")],m:[f,g("minutes")],mm:[f,g("minutes")],H:[f,g("hours")],h:[f,g("hours")],HH:[f,g("hours")],hh:[f,g("hours")],D:[f,g("day")],DD:[k,g("day")],Do:[_,function(p){var C=Y.ordinal,F=p.match(/\d+/);if(this.day=F[0],C)for(var L=1;L<=31;L+=1)C(L).replace(/\[|\]/g,"")===p&&(this.day=L)}],w:[f,g("week")],ww:[k,g("week")],M:[f,g("month")],MM:[k,g("month")],MMM:[_,function(p){var C=P("months"),F=(P("monthsShort")||C.map(function(L){return L.slice(0,3)})).indexOf(p)+1;if(F<1)throw new Error;this.month=F%12||F}],MMMM:[_,function(p){var C=P("months").indexOf(p)+1;if(C<1)throw new Error;this.month=C%12||C}],Y:[/[+-]?\d+/,g("year")],YY:[k,function(p){this.year=S(p)}],YYYY:[/\d{4}/,g("year")],Z:M,ZZ:M};function E(p){var C,F;C=p,F=Y&&Y.formats;for(var L=(p=C.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,function(T,b,m){var w=m&&m.toUpperCase();return b||F[m]||a[m]||F[w].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,function(o,l,h){return l||h.slice(1)})})).match(i),G=L.length,H=0;H<G;H+=1){var Q=L[H],j=B[Q],y=j&&j[0],x=j&&j[1];L[H]=x?{regex:y,parser:x}:Q.replace(/^\[|\]$/g,"")}return function(T){for(var b={},m=0,w=0;m<G;m+=1){var o=L[m];if(typeof o=="string")w+=o.length;else{var l=o.regex,h=o.parser,d=T.slice(w),v=l.exec(d)[0];h.call(b,v),T=T.replace(v,"")}}return function(r){var u=r.afternoon;if(u!==void 0){var e=r.hours;u?e<12&&(r.hours+=12):e===12&&(r.hours=0),delete r.afternoon}}(b),b}}return function(p,C,F){F.p.customParseFormat=!0,p&&p.parseTwoDigitYear&&(S=p.parseTwoDigitYear);var L=C.prototype,G=L.parse;L.parse=function(H){var Q=H.date,j=H.utc,y=H.args;this.$u=j;var x=y[1];if(typeof x=="string"){var T=y[2]===!0,b=y[3]===!0,m=T||b,w=y[2];b&&(w=y[2]),Y=this.$locale(),!T&&w&&(Y=F.Ls[w]),this.$d=function(d,v,r,u){try{if(["x","X"].indexOf(v)>-1)return new Date((v==="X"?1e3:1)*d);var e=E(v)(d),I=e.year,D=e.month,A=e.day,N=e.hours,W=e.minutes,O=e.seconds,$=e.milliseconds,it=e.zone,nt=e.week,dt=new Date,ft=A||(I||D?1:dt.getDate()),ot=I||dt.getFullYear(),z=0;I&&!D||(z=D>0?D-1:dt.getMonth());var Z,q=N||0,st=W||0,K=O||0,rt=$||0;return it?new Date(Date.UTC(ot,z,ft,q,st,K,rt+60*it.offset*1e3)):r?new Date(Date.UTC(ot,z,ft,q,st,K,rt)):(Z=new Date(ot,z,ft,q,st,K,rt),nt&&(Z=u(Z).week(nt).toDate()),Z)}catch{return new Date("")}}(Q,x,j,F),this.init(),w&&w!==!0&&(this.$L=this.locale(w).$L),m&&Q!=this.format(x)&&(this.$d=new Date("")),Y={}}else if(x instanceof Array)for(var o=x.length,l=1;l<=o;l+=1){y[1]=x[l-1];var h=F.apply(this,y);if(h.isValid()){this.$d=h.$d,this.$L=h.$L,this.init();break}l===o&&(this.$d=new Date(""))}else G.call(this,H)}}})}(vt)),vt.exports}var Ne=Re();const Be=At(Ne);var xt={exports:{}},qe=xt.exports,Kt;function Ge(){return Kt||(Kt=1,function(t,s){(function(a,i){t.exports=i()})(qe,function(){return function(a,i){var n=i.prototype,k=n.format;n.format=function(f){var _=this,Y=this.$locale();if(!this.isValid())return k.bind(this)(f);var S=this.$utils(),g=(f||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,function(M){switch(M){case"Q":return Math.ceil((_.$M+1)/3);case"Do":return Y.ordinal(_.$D);case"gggg":return _.weekYear();case"GGGG":return _.isoWeekYear();case"wo":return Y.ordinal(_.week(),"W");case"w":case"ww":return S.s(_.week(),M==="w"?1:2,"0");case"W":case"WW":return S.s(_.isoWeek(),M==="W"?1:2,"0");case"k":case"kk":return S.s(String(_.$H===0?24:_.$H),M==="k"?1:2,"0");case"X":return Math.floor(_.$d.getTime()/1e3);case"x":return _.$d.getTime();case"z":return"["+_.offsetName()+"]";case"zzz":return"["+_.offsetName("long")+"]";default:return M}});return k.bind(this)(g)}}})}(xt)),xt.exports}var He=Ge();const Xe=At(He);var Et=function(){var t=c(function(w,o,l,h){for(l=l||{},h=w.length;h--;l[w[h]]=o);return l},"o"),s=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],a=[1,26],i=[1,27],n=[1,28],k=[1,29],f=[1,30],_=[1,31],Y=[1,32],S=[1,33],g=[1,34],M=[1,9],P=[1,10],V=[1,11],B=[1,12],E=[1,13],p=[1,14],C=[1,15],F=[1,16],L=[1,19],G=[1,20],H=[1,21],Q=[1,22],j=[1,23],y=[1,25],x=[1,35],T={trace:c(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:c(function(o,l,h,d,v,r,u){var e=r.length-1;switch(v){case 1:return r[e-1];case 2:this.$=[];break;case 3:r[e-1].push(r[e]),this.$=r[e-1];break;case 4:case 5:this.$=r[e];break;case 6:case 7:this.$=[];break;case 8:d.setWeekday("monday");break;case 9:d.setWeekday("tuesday");break;case 10:d.setWeekday("wednesday");break;case 11:d.setWeekday("thursday");break;case 12:d.setWeekday("friday");break;case 13:d.setWeekday("saturday");break;case 14:d.setWeekday("sunday");break;case 15:d.setWeekend("friday");break;case 16:d.setWeekend("saturday");break;case 17:d.setDateFormat(r[e].substr(11)),this.$=r[e].substr(11);break;case 18:d.enableInclusiveEndDates(),this.$=r[e].substr(18);break;case 19:d.TopAxis(),this.$=r[e].substr(8);break;case 20:d.setAxisFormat(r[e].substr(11)),this.$=r[e].substr(11);break;case 21:d.setTickInterval(r[e].substr(13)),this.$=r[e].substr(13);break;case 22:d.setExcludes(r[e].substr(9)),this.$=r[e].substr(9);break;case 23:d.setIncludes(r[e].substr(9)),this.$=r[e].substr(9);break;case 24:d.setTodayMarker(r[e].substr(12)),this.$=r[e].substr(12);break;case 27:d.setDiagramTitle(r[e].substr(6)),this.$=r[e].substr(6);break;case 28:this.$=r[e].trim(),d.setAccTitle(this.$);break;case 29:case 30:this.$=r[e].trim(),d.setAccDescription(this.$);break;case 31:d.addSection(r[e].substr(8)),this.$=r[e].substr(8);break;case 33:d.addTask(r[e-1],r[e]),this.$="task";break;case 34:this.$=r[e-1],d.setClickEvent(r[e-1],r[e],null);break;case 35:this.$=r[e-2],d.setClickEvent(r[e-2],r[e-1],r[e]);break;case 36:this.$=r[e-2],d.setClickEvent(r[e-2],r[e-1],null),d.setLink(r[e-2],r[e]);break;case 37:this.$=r[e-3],d.setClickEvent(r[e-3],r[e-2],r[e-1]),d.setLink(r[e-3],r[e]);break;case 38:this.$=r[e-2],d.setClickEvent(r[e-2],r[e],null),d.setLink(r[e-2],r[e-1]);break;case 39:this.$=r[e-3],d.setClickEvent(r[e-3],r[e-1],r[e]),d.setLink(r[e-3],r[e-2]);break;case 40:this.$=r[e-1],d.setLink(r[e-1],r[e]);break;case 41:case 47:this.$=r[e-1]+" "+r[e];break;case 42:case 43:case 45:this.$=r[e-2]+" "+r[e-1]+" "+r[e];break;case 44:case 46:this.$=r[e-3]+" "+r[e-2]+" "+r[e-1]+" "+r[e];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(s,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:a,13:i,14:n,15:k,16:f,17:_,18:Y,19:18,20:S,21:g,22:M,23:P,24:V,25:B,26:E,27:p,28:C,29:F,30:L,31:G,33:H,35:Q,36:j,37:24,38:y,40:x},t(s,[2,7],{1:[2,1]}),t(s,[2,3]),{9:36,11:17,12:a,13:i,14:n,15:k,16:f,17:_,18:Y,19:18,20:S,21:g,22:M,23:P,24:V,25:B,26:E,27:p,28:C,29:F,30:L,31:G,33:H,35:Q,36:j,37:24,38:y,40:x},t(s,[2,5]),t(s,[2,6]),t(s,[2,17]),t(s,[2,18]),t(s,[2,19]),t(s,[2,20]),t(s,[2,21]),t(s,[2,22]),t(s,[2,23]),t(s,[2,24]),t(s,[2,25]),t(s,[2,26]),t(s,[2,27]),{32:[1,37]},{34:[1,38]},t(s,[2,30]),t(s,[2,31]),t(s,[2,32]),{39:[1,39]},t(s,[2,8]),t(s,[2,9]),t(s,[2,10]),t(s,[2,11]),t(s,[2,12]),t(s,[2,13]),t(s,[2,14]),t(s,[2,15]),t(s,[2,16]),{41:[1,40],43:[1,41]},t(s,[2,4]),t(s,[2,28]),t(s,[2,29]),t(s,[2,33]),t(s,[2,34],{42:[1,42],43:[1,43]}),t(s,[2,40],{41:[1,44]}),t(s,[2,35],{43:[1,45]}),t(s,[2,36]),t(s,[2,38],{42:[1,46]}),t(s,[2,37]),t(s,[2,39])],defaultActions:{},parseError:c(function(o,l){if(l.recoverable)this.trace(o);else{var h=new Error(o);throw h.hash=l,h}},"parseError"),parse:c(function(o){var l=this,h=[0],d=[],v=[null],r=[],u=this.table,e="",I=0,D=0,A=2,N=1,W=r.slice.call(arguments,1),O=Object.create(this.lexer),$={yy:{}};for(var it in this.yy)Object.prototype.hasOwnProperty.call(this.yy,it)&&($.yy[it]=this.yy[it]);O.setInput(o,$.yy),$.yy.lexer=O,$.yy.parser=this,typeof O.yylloc>"u"&&(O.yylloc={});var nt=O.yylloc;r.push(nt);var dt=O.options&&O.options.ranges;typeof $.yy.parseError=="function"?this.parseError=$.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function ft(U){h.length=h.length-2*U,v.length=v.length-U,r.length=r.length-U}c(ft,"popStack");function ot(){var U;return U=d.pop()||O.lex()||N,typeof U!="number"&&(U instanceof Array&&(d=U,U=d.pop()),U=l.symbols_[U]||U),U}c(ot,"lex");for(var z,Z,q,st,K={},rt,J,Bt,yt;;){if(Z=h[h.length-1],this.defaultActions[Z]?q=this.defaultActions[Z]:((z===null||typeof z>"u")&&(z=ot()),q=u[Z]&&u[Z][z]),typeof q>"u"||!q.length||!q[0]){var Ct="";yt=[];for(rt in u[Z])this.terminals_[rt]&&rt>A&&yt.push("'"+this.terminals_[rt]+"'");O.showPosition?Ct="Parse error on line "+(I+1)+`:
`+O.showPosition()+`
Expecting `+yt.join(", ")+", got '"+(this.terminals_[z]||z)+"'":Ct="Parse error on line "+(I+1)+": Unexpected "+(z==N?"end of input":"'"+(this.terminals_[z]||z)+"'"),this.parseError(Ct,{text:O.match,token:this.terminals_[z]||z,line:O.yylineno,loc:nt,expected:yt})}if(q[0]instanceof Array&&q.length>1)throw new Error("Parse Error: multiple actions possible at state: "+Z+", token: "+z);switch(q[0]){case 1:h.push(z),v.push(O.yytext),r.push(O.yylloc),h.push(q[1]),z=null,D=O.yyleng,e=O.yytext,I=O.yylineno,nt=O.yylloc;break;case 2:if(J=this.productions_[q[1]][1],K.$=v[v.length-J],K._$={first_line:r[r.length-(J||1)].first_line,last_line:r[r.length-1].last_line,first_column:r[r.length-(J||1)].first_column,last_column:r[r.length-1].last_column},dt&&(K._$.range=[r[r.length-(J||1)].range[0],r[r.length-1].range[1]]),st=this.performAction.apply(K,[e,D,I,$.yy,q[1],v,r].concat(W)),typeof st<"u")return st;J&&(h=h.slice(0,-1*J*2),v=v.slice(0,-1*J),r=r.slice(0,-1*J)),h.push(this.productions_[q[1]][0]),v.push(K.$),r.push(K._$),Bt=u[h[h.length-2]][h[h.length-1]],h.push(Bt);break;case 3:return!0}}return!0},"parse")},b=function(){var w={EOF:1,parseError:c(function(l,h){if(this.yy.parser)this.yy.parser.parseError(l,h);else throw new Error(l)},"parseError"),setInput:c(function(o,l){return this.yy=l||this.yy||{},this._input=o,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:c(function(){var o=this._input[0];this.yytext+=o,this.yyleng++,this.offset++,this.match+=o,this.matched+=o;var l=o.match(/(?:\r\n?|\n).*/g);return l?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),o},"input"),unput:c(function(o){var l=o.length,h=o.split(/(?:\r\n?|\n)/g);this._input=o+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-l),this.offset-=l;var d=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),h.length-1&&(this.yylineno-=h.length-1);var v=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:h?(h.length===d.length?this.yylloc.first_column:0)+d[d.length-h.length].length-h[0].length:this.yylloc.first_column-l},this.options.ranges&&(this.yylloc.range=[v[0],v[0]+this.yyleng-l]),this.yyleng=this.yytext.length,this},"unput"),more:c(function(){return this._more=!0,this},"more"),reject:c(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:c(function(o){this.unput(this.match.slice(o))},"less"),pastInput:c(function(){var o=this.matched.substr(0,this.matched.length-this.match.length);return(o.length>20?"...":"")+o.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:c(function(){var o=this.match;return o.length<20&&(o+=this._input.substr(0,20-o.length)),(o.substr(0,20)+(o.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:c(function(){var o=this.pastInput(),l=new Array(o.length+1).join("-");return o+this.upcomingInput()+`
`+l+"^"},"showPosition"),test_match:c(function(o,l){var h,d,v;if(this.options.backtrack_lexer&&(v={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(v.yylloc.range=this.yylloc.range.slice(0))),d=o[0].match(/(?:\r\n?|\n).*/g),d&&(this.yylineno+=d.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:d?d[d.length-1].length-d[d.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+o[0].length},this.yytext+=o[0],this.match+=o[0],this.matches=o,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(o[0].length),this.matched+=o[0],h=this.performAction.call(this,this.yy,this,l,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),h)return h;if(this._backtrack){for(var r in v)this[r]=v[r];return!1}return!1},"test_match"),next:c(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var o,l,h,d;this._more||(this.yytext="",this.match="");for(var v=this._currentRules(),r=0;r<v.length;r++)if(h=this._input.match(this.rules[v[r]]),h&&(!l||h[0].length>l[0].length)){if(l=h,d=r,this.options.backtrack_lexer){if(o=this.test_match(h,v[r]),o!==!1)return o;if(this._backtrack){l=!1;continue}else return!1}else if(!this.options.flex)break}return l?(o=this.test_match(l,v[d]),o!==!1?o:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:c(function(){var l=this.next();return l||this.lex()},"lex"),begin:c(function(l){this.conditionStack.push(l)},"begin"),popState:c(function(){var l=this.conditionStack.length-1;return l>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:c(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:c(function(l){return l=this.conditionStack.length-1-Math.abs(l||0),l>=0?this.conditionStack[l]:"INITIAL"},"topState"),pushState:c(function(l){this.begin(l)},"pushState"),stateStackSize:c(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:c(function(l,h,d,v){switch(d){case 0:return this.begin("open_directive"),"open_directive";case 1:return this.begin("acc_title"),31;case 2:return this.popState(),"acc_title_value";case 3:return this.begin("acc_descr"),33;case 4:return this.popState(),"acc_descr_value";case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}};return w}();T.lexer=b;function m(){this.yy={}}return c(m,"Parser"),m.prototype=T,T.Parser=m,new m}();Et.parser=Et;var je=Et;X.extend(Ve);X.extend(Be);X.extend(Xe);var Jt={friday:5,saturday:6},tt="",Ft="",Lt=void 0,Yt="",ht=[],kt=[],Wt=new Map,Ot=[],_t=[],ut="",Pt="",re=["active","done","crit","milestone"],Vt=[],mt=!1,zt=!1,Rt="sunday",Dt="saturday",St=0,Ue=c(function(){Ot=[],_t=[],ut="",Vt=[],Tt=0,It=void 0,bt=void 0,R=[],tt="",Ft="",Pt="",Lt=void 0,Yt="",ht=[],kt=[],mt=!1,zt=!1,St=0,Wt=new Map,Le(),Rt="sunday",Dt="saturday"},"clear"),Ze=c(function(t){Ft=t},"setAxisFormat"),Qe=c(function(){return Ft},"getAxisFormat"),$e=c(function(t){Lt=t},"setTickInterval"),Ke=c(function(){return Lt},"getTickInterval"),Je=c(function(t){Yt=t},"setTodayMarker"),tr=c(function(){return Yt},"getTodayMarker"),er=c(function(t){tt=t},"setDateFormat"),rr=c(function(){mt=!0},"enableInclusiveEndDates"),sr=c(function(){return mt},"endDatesAreInclusive"),ar=c(function(){zt=!0},"enableTopAxis"),ir=c(function(){return zt},"topAxisEnabled"),nr=c(function(t){Pt=t},"setDisplayMode"),or=c(function(){return Pt},"getDisplayMode"),cr=c(function(){return tt},"getDateFormat"),lr=c(function(t){ht=t.toLowerCase().split(/[\s,]+/)},"setIncludes"),ur=c(function(){return ht},"getIncludes"),dr=c(function(t){kt=t.toLowerCase().split(/[\s,]+/)},"setExcludes"),fr=c(function(){return kt},"getExcludes"),hr=c(function(){return Wt},"getLinks"),kr=c(function(t){ut=t,Ot.push(t)},"addSection"),mr=c(function(){return Ot},"getSections"),yr=c(function(){let t=te();const s=10;let a=0;for(;!t&&a<s;)t=te(),a++;return _t=R,_t},"getTasks"),se=c(function(t,s,a,i){return i.includes(t.format(s.trim()))?!1:a.includes("weekends")&&(t.isoWeekday()===Jt[Dt]||t.isoWeekday()===Jt[Dt]+1)||a.includes(t.format("dddd").toLowerCase())?!0:a.includes(t.format(s.trim()))},"isInvalidDate"),gr=c(function(t){Rt=t},"setWeekday"),pr=c(function(){return Rt},"getWeekday"),vr=c(function(t){Dt=t},"setWeekend"),ae=c(function(t,s,a,i){if(!a.length||t.manualEndTime)return;let n;t.startTime instanceof Date?n=X(t.startTime):n=X(t.startTime,s,!0),n=n.add(1,"d");let k;t.endTime instanceof Date?k=X(t.endTime):k=X(t.endTime,s,!0);const[f,_]=xr(n,k,s,a,i);t.endTime=f.toDate(),t.renderEndTime=_},"checkTaskDates"),xr=c(function(t,s,a,i,n){let k=!1,f=null;for(;t<=s;)k||(f=s.toDate()),k=se(t,a,i,n),k&&(s=s.add(1,"d")),t=t.add(1,"d");return[s,f]},"fixTaskDates"),Mt=c(function(t,s,a){a=a.trim();const n=/^after\s+(?<ids>[\d\w- ]+)/.exec(a);if(n!==null){let f=null;for(const Y of n.groups.ids.split(" ")){let S=at(Y);S!==void 0&&(!f||S.endTime>f.endTime)&&(f=S)}if(f)return f.endTime;const _=new Date;return _.setHours(0,0,0,0),_}let k=X(a,s.trim(),!0);if(k.isValid())return k.toDate();{wt.debug("Invalid date:"+a),wt.debug("With date format:"+s.trim());const f=new Date(a);if(f===void 0||isNaN(f.getTime())||f.getFullYear()<-1e4||f.getFullYear()>1e4)throw new Error("Invalid date:"+a);return f}},"getStartDate"),ie=c(function(t){const s=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(t.trim());return s!==null?[Number.parseFloat(s[1]),s[2]]:[NaN,"ms"]},"parseDuration"),ne=c(function(t,s,a,i=!1){a=a.trim();const k=/^until\s+(?<ids>[\d\w- ]+)/.exec(a);if(k!==null){let g=null;for(const P of k.groups.ids.split(" ")){let V=at(P);V!==void 0&&(!g||V.startTime<g.startTime)&&(g=V)}if(g)return g.startTime;const M=new Date;return M.setHours(0,0,0,0),M}let f=X(a,s.trim(),!0);if(f.isValid())return i&&(f=f.add(1,"d")),f.toDate();let _=X(t);const[Y,S]=ie(a);if(!Number.isNaN(Y)){const g=_.add(Y,S);g.isValid()&&(_=g)}return _.toDate()},"getEndDate"),Tt=0,lt=c(function(t){return t===void 0?(Tt=Tt+1,"task"+Tt):t},"parseId"),Tr=c(function(t,s){let a;s.substr(0,1)===":"?a=s.substr(1,s.length):a=s;const i=a.split(","),n={};Nt(i,n,re);for(let f=0;f<i.length;f++)i[f]=i[f].trim();let k="";switch(i.length){case 1:n.id=lt(),n.startTime=t.endTime,k=i[0];break;case 2:n.id=lt(),n.startTime=Mt(void 0,tt,i[0]),k=i[1];break;case 3:n.id=lt(i[0]),n.startTime=Mt(void 0,tt,i[1]),k=i[2];break}return k&&(n.endTime=ne(n.startTime,tt,k,mt),n.manualEndTime=X(k,"YYYY-MM-DD",!0).isValid(),ae(n,tt,kt,ht)),n},"compileData"),br=c(function(t,s){let a;s.substr(0,1)===":"?a=s.substr(1,s.length):a=s;const i=a.split(","),n={};Nt(i,n,re);for(let k=0;k<i.length;k++)i[k]=i[k].trim();switch(i.length){case 1:n.id=lt(),n.startTime={type:"prevTaskEnd",id:t},n.endTime={data:i[0]};break;case 2:n.id=lt(),n.startTime={type:"getStartDate",startData:i[0]},n.endTime={data:i[1]};break;case 3:n.id=lt(i[0]),n.startTime={type:"getStartDate",startData:i[1]},n.endTime={data:i[2]};break}return n},"parseData"),It,bt,R=[],oe={},wr=c(function(t,s){const a={section:ut,type:ut,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:s},task:t,classes:[]},i=br(bt,s);a.raw.startTime=i.startTime,a.raw.endTime=i.endTime,a.id=i.id,a.prevTaskId=bt,a.active=i.active,a.done=i.done,a.crit=i.crit,a.milestone=i.milestone,a.order=St,St++;const n=R.push(a);bt=a.id,oe[a.id]=n-1},"addTask"),at=c(function(t){const s=oe[t];return R[s]},"findTaskById"),_r=c(function(t,s){const a={section:ut,type:ut,description:t,task:t,classes:[]},i=Tr(It,s);a.startTime=i.startTime,a.endTime=i.endTime,a.id=i.id,a.active=i.active,a.done=i.done,a.crit=i.crit,a.milestone=i.milestone,It=a,_t.push(a)},"addTaskOrg"),te=c(function(){const t=c(function(a){const i=R[a];let n="";switch(R[a].raw.startTime.type){case"prevTaskEnd":{const k=at(i.prevTaskId);i.startTime=k.endTime;break}case"getStartDate":n=Mt(void 0,tt,R[a].raw.startTime.startData),n&&(R[a].startTime=n);break}return R[a].startTime&&(R[a].endTime=ne(R[a].startTime,tt,R[a].raw.endTime.data,mt),R[a].endTime&&(R[a].processed=!0,R[a].manualEndTime=X(R[a].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),ae(R[a],tt,kt,ht))),R[a].processed},"compileTask");let s=!0;for(const[a,i]of R.entries())t(a),s=s&&i.processed;return s},"compileTasks"),Dr=c(function(t,s){let a=s;ct().securityLevel!=="loose"&&(a=Fe.sanitizeUrl(s)),t.split(",").forEach(function(i){at(i)!==void 0&&(le(i,()=>{window.open(a,"_self")}),Wt.set(i,a))}),ce(t,"clickable")},"setLink"),ce=c(function(t,s){t.split(",").forEach(function(a){let i=at(a);i!==void 0&&i.classes.push(s)})},"setClass"),Cr=c(function(t,s,a){if(ct().securityLevel!=="loose"||s===void 0)return;let i=[];if(typeof a=="string"){i=a.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let k=0;k<i.length;k++){let f=i[k].trim();f.startsWith('"')&&f.endsWith('"')&&(f=f.substr(1,f.length-2)),i[k]=f}}i.length===0&&i.push(t),at(t)!==void 0&&le(t,()=>{Ye.runFunc(s,...i)})},"setClickFun"),le=c(function(t,s){Vt.push(function(){const a=document.querySelector(`[id="${t}"]`);a!==null&&a.addEventListener("click",function(){s()})},function(){const a=document.querySelector(`[id="${t}-text"]`);a!==null&&a.addEventListener("click",function(){s()})})},"pushFun"),Er=c(function(t,s,a){t.split(",").forEach(function(i){Cr(i,s,a)}),ce(t,"clickable")},"setClickEvent"),Sr=c(function(t){Vt.forEach(function(s){s(t)})},"bindFunctions"),Mr={getConfig:c(()=>ct().gantt,"getConfig"),clear:Ue,setDateFormat:er,getDateFormat:cr,enableInclusiveEndDates:rr,endDatesAreInclusive:sr,enableTopAxis:ar,topAxisEnabled:ir,setAxisFormat:Ze,getAxisFormat:Qe,setTickInterval:$e,getTickInterval:Ke,setTodayMarker:Je,getTodayMarker:tr,setAccTitle:me,getAccTitle:ke,setDiagramTitle:he,getDiagramTitle:fe,setDisplayMode:nr,getDisplayMode:or,setAccDescription:de,getAccDescription:ue,addSection:kr,getSections:mr,getTasks:yr,addTask:wr,findTaskById:at,addTaskOrg:_r,setIncludes:lr,getIncludes:ur,setExcludes:dr,getExcludes:fr,setClickEvent:Er,setLink:Dr,getLinks:hr,bindFunctions:Sr,parseDuration:ie,isInvalidDate:se,setWeekday:gr,getWeekday:pr,setWeekend:vr};function Nt(t,s,a){let i=!0;for(;i;)i=!1,a.forEach(function(n){const k="^\\s*"+n+"\\s*$",f=new RegExp(k);t[0].match(f)&&(s[n]=!0,t.shift(1),i=!0)})}c(Nt,"getTaskTags");var Ir=c(function(){wt.debug("Something is calling, setConf, remove the call")},"setConf"),ee={monday:Me,tuesday:Se,wednesday:Ee,thursday:Ce,friday:De,saturday:_e,sunday:we},Ar=c((t,s)=>{let a=[...t].map(()=>-1/0),i=[...t].sort((k,f)=>k.startTime-f.startTime||k.order-f.order),n=0;for(const k of i)for(let f=0;f<a.length;f++)if(k.startTime>=a[f]){a[f]=k.endTime,k.order=f+s,f>n&&(n=f);break}return n},"getMaxIntersections"),et,Fr=c(function(t,s,a,i){const n=ct().gantt,k=ct().securityLevel;let f;k==="sandbox"&&(f=gt("#i"+s));const _=k==="sandbox"?gt(f.nodes()[0].contentDocument.body):gt("body"),Y=k==="sandbox"?f.nodes()[0].contentDocument:document,S=Y.getElementById(s);et=S.parentElement.offsetWidth,et===void 0&&(et=1200),n.useWidth!==void 0&&(et=n.useWidth);const g=i.db.getTasks();let M=[];for(const y of g)M.push(y.type);M=j(M);const P={};let V=2*n.topPadding;if(i.db.getDisplayMode()==="compact"||n.displayMode==="compact"){const y={};for(const T of g)y[T.section]===void 0?y[T.section]=[T]:y[T.section].push(T);let x=0;for(const T of Object.keys(y)){const b=Ar(y[T],x)+1;x+=b,V+=b*(n.barHeight+n.barGap),P[T]=b}}else{V+=g.length*(n.barHeight+n.barGap);for(const y of M)P[y]=g.filter(x=>x.type===y).length}S.setAttribute("viewBox","0 0 "+et+" "+V);const B=_.select(`[id="${s}"]`),E=ye().domain([ge(g,function(y){return y.startTime}),pe(g,function(y){return y.endTime})]).rangeRound([0,et-n.leftPadding-n.rightPadding]);function p(y,x){const T=y.startTime,b=x.startTime;let m=0;return T>b?m=1:T<b&&(m=-1),m}c(p,"taskCompare"),g.sort(p),C(g,et,V),ve(B,V,et,n.useMaxWidth),B.append("text").text(i.db.getDiagramTitle()).attr("x",et/2).attr("y",n.titleTopMargin).attr("class","titleText");function C(y,x,T){const b=n.barHeight,m=b+n.barGap,w=n.topPadding,o=n.leftPadding,l=xe().domain([0,M.length]).range(["#00B9FA","#F95002"]).interpolate(Te);L(m,w,o,x,T,y,i.db.getExcludes(),i.db.getIncludes()),G(o,w,x,T),F(y,m,w,o,b,l,x),H(m,w),Q(o,w,x,T)}c(C,"makeGantt");function F(y,x,T,b,m,w,o){const h=[...new Set(y.map(u=>u.order))].map(u=>y.find(e=>e.order===u));B.append("g").selectAll("rect").data(h).enter().append("rect").attr("x",0).attr("y",function(u,e){return e=u.order,e*x+T-2}).attr("width",function(){return o-n.rightPadding/2}).attr("height",x).attr("class",function(u){for(const[e,I]of M.entries())if(u.type===I)return"section section"+e%n.numberSectionStyles;return"section section0"});const d=B.append("g").selectAll("rect").data(y).enter(),v=i.db.getLinks();if(d.append("rect").attr("id",function(u){return u.id}).attr("rx",3).attr("ry",3).attr("x",function(u){return u.milestone?E(u.startTime)+b+.5*(E(u.endTime)-E(u.startTime))-.5*m:E(u.startTime)+b}).attr("y",function(u,e){return e=u.order,e*x+T}).attr("width",function(u){return u.milestone?m:E(u.renderEndTime||u.endTime)-E(u.startTime)}).attr("height",m).attr("transform-origin",function(u,e){return e=u.order,(E(u.startTime)+b+.5*(E(u.endTime)-E(u.startTime))).toString()+"px "+(e*x+T+.5*m).toString()+"px"}).attr("class",function(u){const e="task";let I="";u.classes.length>0&&(I=u.classes.join(" "));let D=0;for(const[N,W]of M.entries())u.type===W&&(D=N%n.numberSectionStyles);let A="";return u.active?u.crit?A+=" activeCrit":A=" active":u.done?u.crit?A=" doneCrit":A=" done":u.crit&&(A+=" crit"),A.length===0&&(A=" task"),u.milestone&&(A=" milestone "+A),A+=D,A+=" "+I,e+A}),d.append("text").attr("id",function(u){return u.id+"-text"}).text(function(u){return u.task}).attr("font-size",n.fontSize).attr("x",function(u){let e=E(u.startTime),I=E(u.renderEndTime||u.endTime);u.milestone&&(e+=.5*(E(u.endTime)-E(u.startTime))-.5*m),u.milestone&&(I=e+m);const D=this.getBBox().width;return D>I-e?I+D+1.5*n.leftPadding>o?e+b-5:I+b+5:(I-e)/2+e+b}).attr("y",function(u,e){return e=u.order,e*x+n.barHeight/2+(n.fontSize/2-2)+T}).attr("text-height",m).attr("class",function(u){const e=E(u.startTime);let I=E(u.endTime);u.milestone&&(I=e+m);const D=this.getBBox().width;let A="";u.classes.length>0&&(A=u.classes.join(" "));let N=0;for(const[O,$]of M.entries())u.type===$&&(N=O%n.numberSectionStyles);let W="";return u.active&&(u.crit?W="activeCritText"+N:W="activeText"+N),u.done?u.crit?W=W+" doneCritText"+N:W=W+" doneText"+N:u.crit&&(W=W+" critText"+N),u.milestone&&(W+=" milestoneText"),D>I-e?I+D+1.5*n.leftPadding>o?A+" taskTextOutsideLeft taskTextOutside"+N+" "+W:A+" taskTextOutsideRight taskTextOutside"+N+" "+W+" width-"+D:A+" taskText taskText"+N+" "+W+" width-"+D}),ct().securityLevel==="sandbox"){let u;u=gt("#i"+s);const e=u.nodes()[0].contentDocument;d.filter(function(I){return v.has(I.id)}).each(function(I){var D=e.querySelector("#"+I.id),A=e.querySelector("#"+I.id+"-text");const N=D.parentNode;var W=e.createElement("a");W.setAttribute("xlink:href",v.get(I.id)),W.setAttribute("target","_top"),N.appendChild(W),W.appendChild(D),W.appendChild(A)})}}c(F,"drawRects");function L(y,x,T,b,m,w,o,l){if(o.length===0&&l.length===0)return;let h,d;for(const{startTime:D,endTime:A}of w)(h===void 0||D<h)&&(h=D),(d===void 0||A>d)&&(d=A);if(!h||!d)return;if(X(d).diff(X(h),"year")>5){wt.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}const v=i.db.getDateFormat(),r=[];let u=null,e=X(h);for(;e.valueOf()<=d;)i.db.isInvalidDate(e,v,o,l)?u?u.end=e:u={start:e,end:e}:u&&(r.push(u),u=null),e=e.add(1,"d");B.append("g").selectAll("rect").data(r).enter().append("rect").attr("id",function(D){return"exclude-"+D.start.format("YYYY-MM-DD")}).attr("x",function(D){return E(D.start)+T}).attr("y",n.gridLineStartPadding).attr("width",function(D){const A=D.end.add(1,"day");return E(A)-E(D.start)}).attr("height",m-x-n.gridLineStartPadding).attr("transform-origin",function(D,A){return(E(D.start)+T+.5*(E(D.end)-E(D.start))).toString()+"px "+(A*y+.5*m).toString()+"px"}).attr("class","exclude-range")}c(L,"drawExcludeDays");function G(y,x,T,b){let m=be(E).tickSize(-b+x+n.gridLineStartPadding).tickFormat(qt(i.db.getAxisFormat()||n.axisFormat||"%Y-%m-%d"));const o=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(i.db.getTickInterval()||n.tickInterval);if(o!==null){const l=o[1],h=o[2],d=i.db.getWeekday()||n.weekday;switch(h){case"millisecond":m.ticks(Zt.every(l));break;case"second":m.ticks(Ut.every(l));break;case"minute":m.ticks(jt.every(l));break;case"hour":m.ticks(Xt.every(l));break;case"day":m.ticks(Ht.every(l));break;case"week":m.ticks(ee[d].every(l));break;case"month":m.ticks(Gt.every(l));break}}if(B.append("g").attr("class","grid").attr("transform","translate("+y+", "+(b-50)+")").call(m).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),i.db.topAxisEnabled()||n.topAxis){let l=Ie(E).tickSize(-b+x+n.gridLineStartPadding).tickFormat(qt(i.db.getAxisFormat()||n.axisFormat||"%Y-%m-%d"));if(o!==null){const h=o[1],d=o[2],v=i.db.getWeekday()||n.weekday;switch(d){case"millisecond":l.ticks(Zt.every(h));break;case"second":l.ticks(Ut.every(h));break;case"minute":l.ticks(jt.every(h));break;case"hour":l.ticks(Xt.every(h));break;case"day":l.ticks(Ht.every(h));break;case"week":l.ticks(ee[v].every(h));break;case"month":l.ticks(Gt.every(h));break}}B.append("g").attr("class","grid").attr("transform","translate("+y+", "+x+")").call(l).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}c(G,"makeGrid");function H(y,x){let T=0;const b=Object.keys(P).map(m=>[m,P[m]]);B.append("g").selectAll("text").data(b).enter().append(function(m){const w=m[0].split(Ae.lineBreakRegex),o=-(w.length-1)/2,l=Y.createElementNS("http://www.w3.org/2000/svg","text");l.setAttribute("dy",o+"em");for(const[h,d]of w.entries()){const v=Y.createElementNS("http://www.w3.org/2000/svg","tspan");v.setAttribute("alignment-baseline","central"),v.setAttribute("x","10"),h>0&&v.setAttribute("dy","1em"),v.textContent=d,l.appendChild(v)}return l}).attr("x",10).attr("y",function(m,w){if(w>0)for(let o=0;o<w;o++)return T+=b[w-1][1],m[1]*y/2+T*y+x;else return m[1]*y/2+x}).attr("font-size",n.sectionFontSize).attr("class",function(m){for(const[w,o]of M.entries())if(m[0]===o)return"sectionTitle sectionTitle"+w%n.numberSectionStyles;return"sectionTitle"})}c(H,"vertLabels");function Q(y,x,T,b){const m=i.db.getTodayMarker();if(m==="off")return;const w=B.append("g").attr("class","today"),o=new Date,l=w.append("line");l.attr("x1",E(o)+y).attr("x2",E(o)+y).attr("y1",n.titleTopMargin).attr("y2",b-n.titleTopMargin).attr("class","today"),m!==""&&l.attr("style",m.replace(/,/g,";"))}c(Q,"drawToday");function j(y){const x={},T=[];for(let b=0,m=y.length;b<m;++b)Object.prototype.hasOwnProperty.call(x,y[b])||(x[y[b]]=!0,T.push(y[b]));return T}c(j,"checkUnique")},"draw"),Lr={setConf:Ir,draw:Fr},Yr=c(t=>`
  .mermaid-main-font {
        font-family: ${t.fontFamily};
  }

  .exclude-range {
    fill: ${t.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${t.sectionBkgColor};
  }

  .section2 {
    fill: ${t.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${t.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${t.titleColor};
  }

  .sectionTitle1 {
    fill: ${t.titleColor};
  }

  .sectionTitle2 {
    fill: ${t.titleColor};
  }

  .sectionTitle3 {
    fill: ${t.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${t.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${t.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${t.fontFamily};
    fill: ${t.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${t.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${t.taskTextDarkColor};
    text-anchor: start;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${t.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${t.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${t.taskBkgColor};
    stroke: ${t.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${t.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${t.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${t.activeTaskBkgColor};
    stroke: ${t.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${t.doneTaskBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.doneTaskBkgColor};
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
    fill: ${t.taskTextDarkColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${t.titleColor||t.textColor};
    font-family: ${t.fontFamily};
  }
`,"getStyles"),Wr=Yr,Br={parser:je,db:Mr,renderer:Lr,styles:Wr};export{Br as diagram};

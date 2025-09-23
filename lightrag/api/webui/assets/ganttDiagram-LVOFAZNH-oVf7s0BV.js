import{_ as l,g as ue,s as de,t as fe,q as he,a as ke,b as me,c as ct,d as gt,aF as ye,aG as ge,aH as pe,e as ve,R as xe,aI as Te,aJ as X,l as wt,aK as be,aL as qt,aM as Gt,aN as we,aO as _e,aP as De,aQ as Ce,aR as Se,aS as Ee,aT as Me,aU as Ht,aV as Xt,aW as Ut,aX as jt,aY as Zt,aZ as Ie,k as Ae,j as Le,z as Fe,u as Ye}from"./mermaid-vendor-BVzwgVwp.js";import{g as At}from"./react-vendor-CI9PN-jl.js";import"./feature-graph-CPVE2FPi.js";import"./graph-vendor-C2ay-rh5.js";import"./ui-vendor-BNB5trKt.js";import"./utils-vendor-BaQYYNaI.js";var pt={exports:{}},We=pt.exports,$t;function Oe(){return $t||($t=1,(function(t,a){(function(s,n){t.exports=n()})(We,(function(){var s="day";return function(n,i,h){var f=function(E){return E.add(4-E.isoWeekday(),s)},_=i.prototype;_.isoWeekYear=function(){return f(this).year()},_.isoWeek=function(E){if(!this.$utils().u(E))return this.add(7*(E-this.isoWeek()),s);var p,M,P,V,B=f(this),S=(p=this.isoWeekYear(),M=this.$u,P=(M?h.utc:h)().year(p).startOf("year"),V=4-P.isoWeekday(),P.isoWeekday()>4&&(V+=7),P.add(V,s));return B.diff(S,"week")+1},_.isoWeekday=function(E){return this.$utils().u(E)?this.day()||7:this.day(this.day()%7?E:E-7)};var Y=_.startOf;_.startOf=function(E,p){var M=this.$utils(),P=!!M.u(p)||p;return M.p(E)==="isoweek"?P?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):Y.bind(this)(E,p)}}}))})(pt)),pt.exports}var Pe=Oe();const Ve=At(Pe);var vt={exports:{}},ze=vt.exports,Qt;function Re(){return Qt||(Qt=1,(function(t,a){(function(s,n){t.exports=n()})(ze,(function(){var s={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},n=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,i=/\d/,h=/\d\d/,f=/\d\d?/,_=/\d*[^-_:/,()\s\d]+/,Y={},E=function(v){return(v=+v)+(v>68?1900:2e3)},p=function(v){return function(C){this[v]=+C}},M=[/[+-]\d\d:?(\d\d)?|Z/,function(v){(this.zone||(this.zone={})).offset=(function(C){if(!C||C==="Z")return 0;var L=C.match(/([+-]|\d\d)/g),F=60*L[1]+(+L[2]||0);return F===0?0:L[0]==="+"?-F:F})(v)}],P=function(v){var C=Y[v];return C&&(C.indexOf?C:C.s.concat(C.f))},V=function(v,C){var L,F=Y.meridiem;if(F){for(var G=1;G<=24;G+=1)if(v.indexOf(F(G,0,C))>-1){L=G>12;break}}else L=v===(C?"pm":"PM");return L},B={A:[_,function(v){this.afternoon=V(v,!1)}],a:[_,function(v){this.afternoon=V(v,!0)}],Q:[i,function(v){this.month=3*(v-1)+1}],S:[i,function(v){this.milliseconds=100*+v}],SS:[h,function(v){this.milliseconds=10*+v}],SSS:[/\d{3}/,function(v){this.milliseconds=+v}],s:[f,p("seconds")],ss:[f,p("seconds")],m:[f,p("minutes")],mm:[f,p("minutes")],H:[f,p("hours")],h:[f,p("hours")],HH:[f,p("hours")],hh:[f,p("hours")],D:[f,p("day")],DD:[h,p("day")],Do:[_,function(v){var C=Y.ordinal,L=v.match(/\d+/);if(this.day=L[0],C)for(var F=1;F<=31;F+=1)C(F).replace(/\[|\]/g,"")===v&&(this.day=F)}],w:[f,p("week")],ww:[h,p("week")],M:[f,p("month")],MM:[h,p("month")],MMM:[_,function(v){var C=P("months"),L=(P("monthsShort")||C.map((function(F){return F.slice(0,3)}))).indexOf(v)+1;if(L<1)throw new Error;this.month=L%12||L}],MMMM:[_,function(v){var C=P("months").indexOf(v)+1;if(C<1)throw new Error;this.month=C%12||C}],Y:[/[+-]?\d+/,p("year")],YY:[h,function(v){this.year=E(v)}],YYYY:[/\d{4}/,p("year")],Z:M,ZZ:M};function S(v){var C,L;C=v,L=Y&&Y.formats;for(var F=(v=C.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,(function(b,x,g){var w=g&&g.toUpperCase();return x||L[g]||s[g]||L[w].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,(function(c,d,k){return d||k.slice(1)}))}))).match(n),G=F.length,H=0;H<G;H+=1){var $=F[H],U=B[$],y=U&&U[0],T=U&&U[1];F[H]=T?{regex:y,parser:T}:$.replace(/^\[|\]$/g,"")}return function(b){for(var x={},g=0,w=0;g<G;g+=1){var c=F[g];if(typeof c=="string")w+=c.length;else{var d=c.regex,k=c.parser,u=b.slice(w),m=d.exec(u)[0];k.call(x,m),b=b.replace(m,"")}}return(function(r){var o=r.afternoon;if(o!==void 0){var e=r.hours;o?e<12&&(r.hours+=12):e===12&&(r.hours=0),delete r.afternoon}})(x),x}}return function(v,C,L){L.p.customParseFormat=!0,v&&v.parseTwoDigitYear&&(E=v.parseTwoDigitYear);var F=C.prototype,G=F.parse;F.parse=function(H){var $=H.date,U=H.utc,y=H.args;this.$u=U;var T=y[1];if(typeof T=="string"){var b=y[2]===!0,x=y[3]===!0,g=b||x,w=y[2];x&&(w=y[2]),Y=this.$locale(),!b&&w&&(Y=L.Ls[w]),this.$d=(function(u,m,r,o){try{if(["x","X"].indexOf(m)>-1)return new Date((m==="X"?1e3:1)*u);var e=S(m)(u),I=e.year,D=e.month,A=e.day,N=e.hours,W=e.minutes,O=e.seconds,Q=e.milliseconds,it=e.zone,nt=e.week,dt=new Date,ft=A||(I||D?1:dt.getDate()),ot=I||dt.getFullYear(),z=0;I&&!D||(z=D>0?D-1:dt.getMonth());var Z,q=N||0,st=W||0,K=O||0,rt=Q||0;return it?new Date(Date.UTC(ot,z,ft,q,st,K,rt+60*it.offset*1e3)):r?new Date(Date.UTC(ot,z,ft,q,st,K,rt)):(Z=new Date(ot,z,ft,q,st,K,rt),nt&&(Z=o(Z).week(nt).toDate()),Z)}catch{return new Date("")}})($,T,U,L),this.init(),w&&w!==!0&&(this.$L=this.locale(w).$L),g&&$!=this.format(T)&&(this.$d=new Date("")),Y={}}else if(T instanceof Array)for(var c=T.length,d=1;d<=c;d+=1){y[1]=T[d-1];var k=L.apply(this,y);if(k.isValid()){this.$d=k.$d,this.$L=k.$L,this.init();break}d===c&&(this.$d=new Date(""))}else G.call(this,H)}}}))})(vt)),vt.exports}var Ne=Re();const Be=At(Ne);var xt={exports:{}},qe=xt.exports,Kt;function Ge(){return Kt||(Kt=1,(function(t,a){(function(s,n){t.exports=n()})(qe,(function(){return function(s,n){var i=n.prototype,h=i.format;i.format=function(f){var _=this,Y=this.$locale();if(!this.isValid())return h.bind(this)(f);var E=this.$utils(),p=(f||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,(function(M){switch(M){case"Q":return Math.ceil((_.$M+1)/3);case"Do":return Y.ordinal(_.$D);case"gggg":return _.weekYear();case"GGGG":return _.isoWeekYear();case"wo":return Y.ordinal(_.week(),"W");case"w":case"ww":return E.s(_.week(),M==="w"?1:2,"0");case"W":case"WW":return E.s(_.isoWeek(),M==="W"?1:2,"0");case"k":case"kk":return E.s(String(_.$H===0?24:_.$H),M==="k"?1:2,"0");case"X":return Math.floor(_.$d.getTime()/1e3);case"x":return _.$d.getTime();case"z":return"["+_.offsetName()+"]";case"zzz":return"["+_.offsetName("long")+"]";default:return M}}));return h.bind(this)(p)}}}))})(xt)),xt.exports}var He=Ge();const Xe=At(He);var St=(function(){var t=l(function(w,c,d,k){for(d=d||{},k=w.length;k--;d[w[k]]=c);return d},"o"),a=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],s=[1,26],n=[1,27],i=[1,28],h=[1,29],f=[1,30],_=[1,31],Y=[1,32],E=[1,33],p=[1,34],M=[1,9],P=[1,10],V=[1,11],B=[1,12],S=[1,13],v=[1,14],C=[1,15],L=[1,16],F=[1,19],G=[1,20],H=[1,21],$=[1,22],U=[1,23],y=[1,25],T=[1,35],b={trace:l(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:l(function(c,d,k,u,m,r,o){var e=r.length-1;switch(m){case 1:return r[e-1];case 2:this.$=[];break;case 3:r[e-1].push(r[e]),this.$=r[e-1];break;case 4:case 5:this.$=r[e];break;case 6:case 7:this.$=[];break;case 8:u.setWeekday("monday");break;case 9:u.setWeekday("tuesday");break;case 10:u.setWeekday("wednesday");break;case 11:u.setWeekday("thursday");break;case 12:u.setWeekday("friday");break;case 13:u.setWeekday("saturday");break;case 14:u.setWeekday("sunday");break;case 15:u.setWeekend("friday");break;case 16:u.setWeekend("saturday");break;case 17:u.setDateFormat(r[e].substr(11)),this.$=r[e].substr(11);break;case 18:u.enableInclusiveEndDates(),this.$=r[e].substr(18);break;case 19:u.TopAxis(),this.$=r[e].substr(8);break;case 20:u.setAxisFormat(r[e].substr(11)),this.$=r[e].substr(11);break;case 21:u.setTickInterval(r[e].substr(13)),this.$=r[e].substr(13);break;case 22:u.setExcludes(r[e].substr(9)),this.$=r[e].substr(9);break;case 23:u.setIncludes(r[e].substr(9)),this.$=r[e].substr(9);break;case 24:u.setTodayMarker(r[e].substr(12)),this.$=r[e].substr(12);break;case 27:u.setDiagramTitle(r[e].substr(6)),this.$=r[e].substr(6);break;case 28:this.$=r[e].trim(),u.setAccTitle(this.$);break;case 29:case 30:this.$=r[e].trim(),u.setAccDescription(this.$);break;case 31:u.addSection(r[e].substr(8)),this.$=r[e].substr(8);break;case 33:u.addTask(r[e-1],r[e]),this.$="task";break;case 34:this.$=r[e-1],u.setClickEvent(r[e-1],r[e],null);break;case 35:this.$=r[e-2],u.setClickEvent(r[e-2],r[e-1],r[e]);break;case 36:this.$=r[e-2],u.setClickEvent(r[e-2],r[e-1],null),u.setLink(r[e-2],r[e]);break;case 37:this.$=r[e-3],u.setClickEvent(r[e-3],r[e-2],r[e-1]),u.setLink(r[e-3],r[e]);break;case 38:this.$=r[e-2],u.setClickEvent(r[e-2],r[e],null),u.setLink(r[e-2],r[e-1]);break;case 39:this.$=r[e-3],u.setClickEvent(r[e-3],r[e-1],r[e]),u.setLink(r[e-3],r[e-2]);break;case 40:this.$=r[e-1],u.setLink(r[e-1],r[e]);break;case 41:case 47:this.$=r[e-1]+" "+r[e];break;case 42:case 43:case 45:this.$=r[e-2]+" "+r[e-1]+" "+r[e];break;case 44:case 46:this.$=r[e-3]+" "+r[e-2]+" "+r[e-1]+" "+r[e];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(a,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:s,13:n,14:i,15:h,16:f,17:_,18:Y,19:18,20:E,21:p,22:M,23:P,24:V,25:B,26:S,27:v,28:C,29:L,30:F,31:G,33:H,35:$,36:U,37:24,38:y,40:T},t(a,[2,7],{1:[2,1]}),t(a,[2,3]),{9:36,11:17,12:s,13:n,14:i,15:h,16:f,17:_,18:Y,19:18,20:E,21:p,22:M,23:P,24:V,25:B,26:S,27:v,28:C,29:L,30:F,31:G,33:H,35:$,36:U,37:24,38:y,40:T},t(a,[2,5]),t(a,[2,6]),t(a,[2,17]),t(a,[2,18]),t(a,[2,19]),t(a,[2,20]),t(a,[2,21]),t(a,[2,22]),t(a,[2,23]),t(a,[2,24]),t(a,[2,25]),t(a,[2,26]),t(a,[2,27]),{32:[1,37]},{34:[1,38]},t(a,[2,30]),t(a,[2,31]),t(a,[2,32]),{39:[1,39]},t(a,[2,8]),t(a,[2,9]),t(a,[2,10]),t(a,[2,11]),t(a,[2,12]),t(a,[2,13]),t(a,[2,14]),t(a,[2,15]),t(a,[2,16]),{41:[1,40],43:[1,41]},t(a,[2,4]),t(a,[2,28]),t(a,[2,29]),t(a,[2,33]),t(a,[2,34],{42:[1,42],43:[1,43]}),t(a,[2,40],{41:[1,44]}),t(a,[2,35],{43:[1,45]}),t(a,[2,36]),t(a,[2,38],{42:[1,46]}),t(a,[2,37]),t(a,[2,39])],defaultActions:{},parseError:l(function(c,d){if(d.recoverable)this.trace(c);else{var k=new Error(c);throw k.hash=d,k}},"parseError"),parse:l(function(c){var d=this,k=[0],u=[],m=[null],r=[],o=this.table,e="",I=0,D=0,A=2,N=1,W=r.slice.call(arguments,1),O=Object.create(this.lexer),Q={yy:{}};for(var it in this.yy)Object.prototype.hasOwnProperty.call(this.yy,it)&&(Q.yy[it]=this.yy[it]);O.setInput(c,Q.yy),Q.yy.lexer=O,Q.yy.parser=this,typeof O.yylloc>"u"&&(O.yylloc={});var nt=O.yylloc;r.push(nt);var dt=O.options&&O.options.ranges;typeof Q.yy.parseError=="function"?this.parseError=Q.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function ft(j){k.length=k.length-2*j,m.length=m.length-j,r.length=r.length-j}l(ft,"popStack");function ot(){var j;return j=u.pop()||O.lex()||N,typeof j!="number"&&(j instanceof Array&&(u=j,j=u.pop()),j=d.symbols_[j]||j),j}l(ot,"lex");for(var z,Z,q,st,K={},rt,J,Bt,yt;;){if(Z=k[k.length-1],this.defaultActions[Z]?q=this.defaultActions[Z]:((z===null||typeof z>"u")&&(z=ot()),q=o[Z]&&o[Z][z]),typeof q>"u"||!q.length||!q[0]){var Ct="";yt=[];for(rt in o[Z])this.terminals_[rt]&&rt>A&&yt.push("'"+this.terminals_[rt]+"'");O.showPosition?Ct="Parse error on line "+(I+1)+`:
`+O.showPosition()+`
Expecting `+yt.join(", ")+", got '"+(this.terminals_[z]||z)+"'":Ct="Parse error on line "+(I+1)+": Unexpected "+(z==N?"end of input":"'"+(this.terminals_[z]||z)+"'"),this.parseError(Ct,{text:O.match,token:this.terminals_[z]||z,line:O.yylineno,loc:nt,expected:yt})}if(q[0]instanceof Array&&q.length>1)throw new Error("Parse Error: multiple actions possible at state: "+Z+", token: "+z);switch(q[0]){case 1:k.push(z),m.push(O.yytext),r.push(O.yylloc),k.push(q[1]),z=null,D=O.yyleng,e=O.yytext,I=O.yylineno,nt=O.yylloc;break;case 2:if(J=this.productions_[q[1]][1],K.$=m[m.length-J],K._$={first_line:r[r.length-(J||1)].first_line,last_line:r[r.length-1].last_line,first_column:r[r.length-(J||1)].first_column,last_column:r[r.length-1].last_column},dt&&(K._$.range=[r[r.length-(J||1)].range[0],r[r.length-1].range[1]]),st=this.performAction.apply(K,[e,D,I,Q.yy,q[1],m,r].concat(W)),typeof st<"u")return st;J&&(k=k.slice(0,-1*J*2),m=m.slice(0,-1*J),r=r.slice(0,-1*J)),k.push(this.productions_[q[1]][0]),m.push(K.$),r.push(K._$),Bt=o[k[k.length-2]][k[k.length-1]],k.push(Bt);break;case 3:return!0}}return!0},"parse")},x=(function(){var w={EOF:1,parseError:l(function(d,k){if(this.yy.parser)this.yy.parser.parseError(d,k);else throw new Error(d)},"parseError"),setInput:l(function(c,d){return this.yy=d||this.yy||{},this._input=c,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:l(function(){var c=this._input[0];this.yytext+=c,this.yyleng++,this.offset++,this.match+=c,this.matched+=c;var d=c.match(/(?:\r\n?|\n).*/g);return d?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),c},"input"),unput:l(function(c){var d=c.length,k=c.split(/(?:\r\n?|\n)/g);this._input=c+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-d),this.offset-=d;var u=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),k.length-1&&(this.yylineno-=k.length-1);var m=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:k?(k.length===u.length?this.yylloc.first_column:0)+u[u.length-k.length].length-k[0].length:this.yylloc.first_column-d},this.options.ranges&&(this.yylloc.range=[m[0],m[0]+this.yyleng-d]),this.yyleng=this.yytext.length,this},"unput"),more:l(function(){return this._more=!0,this},"more"),reject:l(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:l(function(c){this.unput(this.match.slice(c))},"less"),pastInput:l(function(){var c=this.matched.substr(0,this.matched.length-this.match.length);return(c.length>20?"...":"")+c.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:l(function(){var c=this.match;return c.length<20&&(c+=this._input.substr(0,20-c.length)),(c.substr(0,20)+(c.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:l(function(){var c=this.pastInput(),d=new Array(c.length+1).join("-");return c+this.upcomingInput()+`
`+d+"^"},"showPosition"),test_match:l(function(c,d){var k,u,m;if(this.options.backtrack_lexer&&(m={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(m.yylloc.range=this.yylloc.range.slice(0))),u=c[0].match(/(?:\r\n?|\n).*/g),u&&(this.yylineno+=u.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:u?u[u.length-1].length-u[u.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+c[0].length},this.yytext+=c[0],this.match+=c[0],this.matches=c,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(c[0].length),this.matched+=c[0],k=this.performAction.call(this,this.yy,this,d,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),k)return k;if(this._backtrack){for(var r in m)this[r]=m[r];return!1}return!1},"test_match"),next:l(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var c,d,k,u;this._more||(this.yytext="",this.match="");for(var m=this._currentRules(),r=0;r<m.length;r++)if(k=this._input.match(this.rules[m[r]]),k&&(!d||k[0].length>d[0].length)){if(d=k,u=r,this.options.backtrack_lexer){if(c=this.test_match(k,m[r]),c!==!1)return c;if(this._backtrack){d=!1;continue}else return!1}else if(!this.options.flex)break}return d?(c=this.test_match(d,m[u]),c!==!1?c:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:l(function(){var d=this.next();return d||this.lex()},"lex"),begin:l(function(d){this.conditionStack.push(d)},"begin"),popState:l(function(){var d=this.conditionStack.length-1;return d>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:l(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:l(function(d){return d=this.conditionStack.length-1-Math.abs(d||0),d>=0?this.conditionStack[d]:"INITIAL"},"topState"),pushState:l(function(d){this.begin(d)},"pushState"),stateStackSize:l(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:l(function(d,k,u,m){switch(u){case 0:return this.begin("open_directive"),"open_directive";case 1:return this.begin("acc_title"),31;case 2:return this.popState(),"acc_title_value";case 3:return this.begin("acc_descr"),33;case 4:return this.popState(),"acc_descr_value";case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}};return w})();b.lexer=x;function g(){this.yy={}}return l(g,"Parser"),g.prototype=b,b.Parser=g,new g})();St.parser=St;var Ue=St;X.extend(Ve);X.extend(Be);X.extend(Xe);var Jt={friday:5,saturday:6},tt="",Lt="",Ft=void 0,Yt="",ht=[],kt=[],Wt=new Map,Ot=[],_t=[],ut="",Pt="",re=["active","done","crit","milestone","vert"],Vt=[],mt=!1,zt=!1,Rt="sunday",Dt="saturday",Et=0,je=l(function(){Ot=[],_t=[],ut="",Vt=[],Tt=0,It=void 0,bt=void 0,R=[],tt="",Lt="",Pt="",Ft=void 0,Yt="",ht=[],kt=[],mt=!1,zt=!1,Et=0,Wt=new Map,Fe(),Rt="sunday",Dt="saturday"},"clear"),Ze=l(function(t){Lt=t},"setAxisFormat"),$e=l(function(){return Lt},"getAxisFormat"),Qe=l(function(t){Ft=t},"setTickInterval"),Ke=l(function(){return Ft},"getTickInterval"),Je=l(function(t){Yt=t},"setTodayMarker"),tr=l(function(){return Yt},"getTodayMarker"),er=l(function(t){tt=t},"setDateFormat"),rr=l(function(){mt=!0},"enableInclusiveEndDates"),sr=l(function(){return mt},"endDatesAreInclusive"),ar=l(function(){zt=!0},"enableTopAxis"),ir=l(function(){return zt},"topAxisEnabled"),nr=l(function(t){Pt=t},"setDisplayMode"),or=l(function(){return Pt},"getDisplayMode"),cr=l(function(){return tt},"getDateFormat"),lr=l(function(t){ht=t.toLowerCase().split(/[\s,]+/)},"setIncludes"),ur=l(function(){return ht},"getIncludes"),dr=l(function(t){kt=t.toLowerCase().split(/[\s,]+/)},"setExcludes"),fr=l(function(){return kt},"getExcludes"),hr=l(function(){return Wt},"getLinks"),kr=l(function(t){ut=t,Ot.push(t)},"addSection"),mr=l(function(){return Ot},"getSections"),yr=l(function(){let t=te();const a=10;let s=0;for(;!t&&s<a;)t=te(),s++;return _t=R,_t},"getTasks"),se=l(function(t,a,s,n){const i=t.format(a.trim()),h=t.format("YYYY-MM-DD");return n.includes(i)||n.includes(h)?!1:s.includes("weekends")&&(t.isoWeekday()===Jt[Dt]||t.isoWeekday()===Jt[Dt]+1)||s.includes(t.format("dddd").toLowerCase())?!0:s.includes(i)||s.includes(h)},"isInvalidDate"),gr=l(function(t){Rt=t},"setWeekday"),pr=l(function(){return Rt},"getWeekday"),vr=l(function(t){Dt=t},"setWeekend"),ae=l(function(t,a,s,n){if(!s.length||t.manualEndTime)return;let i;t.startTime instanceof Date?i=X(t.startTime):i=X(t.startTime,a,!0),i=i.add(1,"d");let h;t.endTime instanceof Date?h=X(t.endTime):h=X(t.endTime,a,!0);const[f,_]=xr(i,h,a,s,n);t.endTime=f.toDate(),t.renderEndTime=_},"checkTaskDates"),xr=l(function(t,a,s,n,i){let h=!1,f=null;for(;t<=a;)h||(f=a.toDate()),h=se(t,s,n,i),h&&(a=a.add(1,"d")),t=t.add(1,"d");return[a,f]},"fixTaskDates"),Mt=l(function(t,a,s){if(s=s.trim(),(a.trim()==="x"||a.trim()==="X")&&/^\d+$/.test(s))return new Date(Number(s));const i=/^after\s+(?<ids>[\d\w- ]+)/.exec(s);if(i!==null){let f=null;for(const Y of i.groups.ids.split(" ")){let E=at(Y);E!==void 0&&(!f||E.endTime>f.endTime)&&(f=E)}if(f)return f.endTime;const _=new Date;return _.setHours(0,0,0,0),_}let h=X(s,a.trim(),!0);if(h.isValid())return h.toDate();{wt.debug("Invalid date:"+s),wt.debug("With date format:"+a.trim());const f=new Date(s);if(f===void 0||isNaN(f.getTime())||f.getFullYear()<-1e4||f.getFullYear()>1e4)throw new Error("Invalid date:"+s);return f}},"getStartDate"),ie=l(function(t){const a=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(t.trim());return a!==null?[Number.parseFloat(a[1]),a[2]]:[NaN,"ms"]},"parseDuration"),ne=l(function(t,a,s,n=!1){s=s.trim();const h=/^until\s+(?<ids>[\d\w- ]+)/.exec(s);if(h!==null){let p=null;for(const P of h.groups.ids.split(" ")){let V=at(P);V!==void 0&&(!p||V.startTime<p.startTime)&&(p=V)}if(p)return p.startTime;const M=new Date;return M.setHours(0,0,0,0),M}let f=X(s,a.trim(),!0);if(f.isValid())return n&&(f=f.add(1,"d")),f.toDate();let _=X(t);const[Y,E]=ie(s);if(!Number.isNaN(Y)){const p=_.add(Y,E);p.isValid()&&(_=p)}return _.toDate()},"getEndDate"),Tt=0,lt=l(function(t){return t===void 0?(Tt=Tt+1,"task"+Tt):t},"parseId"),Tr=l(function(t,a){let s;a.substr(0,1)===":"?s=a.substr(1,a.length):s=a;const n=s.split(","),i={};Nt(n,i,re);for(let f=0;f<n.length;f++)n[f]=n[f].trim();let h="";switch(n.length){case 1:i.id=lt(),i.startTime=t.endTime,h=n[0];break;case 2:i.id=lt(),i.startTime=Mt(void 0,tt,n[0]),h=n[1];break;case 3:i.id=lt(n[0]),i.startTime=Mt(void 0,tt,n[1]),h=n[2];break}return h&&(i.endTime=ne(i.startTime,tt,h,mt),i.manualEndTime=X(h,"YYYY-MM-DD",!0).isValid(),ae(i,tt,kt,ht)),i},"compileData"),br=l(function(t,a){let s;a.substr(0,1)===":"?s=a.substr(1,a.length):s=a;const n=s.split(","),i={};Nt(n,i,re);for(let h=0;h<n.length;h++)n[h]=n[h].trim();switch(n.length){case 1:i.id=lt(),i.startTime={type:"prevTaskEnd",id:t},i.endTime={data:n[0]};break;case 2:i.id=lt(),i.startTime={type:"getStartDate",startData:n[0]},i.endTime={data:n[1]};break;case 3:i.id=lt(n[0]),i.startTime={type:"getStartDate",startData:n[1]},i.endTime={data:n[2]};break}return i},"parseData"),It,bt,R=[],oe={},wr=l(function(t,a){const s={section:ut,type:ut,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:a},task:t,classes:[]},n=br(bt,a);s.raw.startTime=n.startTime,s.raw.endTime=n.endTime,s.id=n.id,s.prevTaskId=bt,s.active=n.active,s.done=n.done,s.crit=n.crit,s.milestone=n.milestone,s.vert=n.vert,s.order=Et,Et++;const i=R.push(s);bt=s.id,oe[s.id]=i-1},"addTask"),at=l(function(t){const a=oe[t];return R[a]},"findTaskById"),_r=l(function(t,a){const s={section:ut,type:ut,description:t,task:t,classes:[]},n=Tr(It,a);s.startTime=n.startTime,s.endTime=n.endTime,s.id=n.id,s.active=n.active,s.done=n.done,s.crit=n.crit,s.milestone=n.milestone,s.vert=n.vert,It=s,_t.push(s)},"addTaskOrg"),te=l(function(){const t=l(function(s){const n=R[s];let i="";switch(R[s].raw.startTime.type){case"prevTaskEnd":{const h=at(n.prevTaskId);n.startTime=h.endTime;break}case"getStartDate":i=Mt(void 0,tt,R[s].raw.startTime.startData),i&&(R[s].startTime=i);break}return R[s].startTime&&(R[s].endTime=ne(R[s].startTime,tt,R[s].raw.endTime.data,mt),R[s].endTime&&(R[s].processed=!0,R[s].manualEndTime=X(R[s].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),ae(R[s],tt,kt,ht))),R[s].processed},"compileTask");let a=!0;for(const[s,n]of R.entries())t(s),a=a&&n.processed;return a},"compileTasks"),Dr=l(function(t,a){let s=a;ct().securityLevel!=="loose"&&(s=Le.sanitizeUrl(a)),t.split(",").forEach(function(n){at(n)!==void 0&&(le(n,()=>{window.open(s,"_self")}),Wt.set(n,s))}),ce(t,"clickable")},"setLink"),ce=l(function(t,a){t.split(",").forEach(function(s){let n=at(s);n!==void 0&&n.classes.push(a)})},"setClass"),Cr=l(function(t,a,s){if(ct().securityLevel!=="loose"||a===void 0)return;let n=[];if(typeof s=="string"){n=s.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let h=0;h<n.length;h++){let f=n[h].trim();f.startsWith('"')&&f.endsWith('"')&&(f=f.substr(1,f.length-2)),n[h]=f}}n.length===0&&n.push(t),at(t)!==void 0&&le(t,()=>{Ye.runFunc(a,...n)})},"setClickFun"),le=l(function(t,a){Vt.push(function(){const s=document.querySelector(`[id="${t}"]`);s!==null&&s.addEventListener("click",function(){a()})},function(){const s=document.querySelector(`[id="${t}-text"]`);s!==null&&s.addEventListener("click",function(){a()})})},"pushFun"),Sr=l(function(t,a,s){t.split(",").forEach(function(n){Cr(n,a,s)}),ce(t,"clickable")},"setClickEvent"),Er=l(function(t){Vt.forEach(function(a){a(t)})},"bindFunctions"),Mr={getConfig:l(()=>ct().gantt,"getConfig"),clear:je,setDateFormat:er,getDateFormat:cr,enableInclusiveEndDates:rr,endDatesAreInclusive:sr,enableTopAxis:ar,topAxisEnabled:ir,setAxisFormat:Ze,getAxisFormat:$e,setTickInterval:Qe,getTickInterval:Ke,setTodayMarker:Je,getTodayMarker:tr,setAccTitle:me,getAccTitle:ke,setDiagramTitle:he,getDiagramTitle:fe,setDisplayMode:nr,getDisplayMode:or,setAccDescription:de,getAccDescription:ue,addSection:kr,getSections:mr,getTasks:yr,addTask:wr,findTaskById:at,addTaskOrg:_r,setIncludes:lr,getIncludes:ur,setExcludes:dr,getExcludes:fr,setClickEvent:Sr,setLink:Dr,getLinks:hr,bindFunctions:Er,parseDuration:ie,isInvalidDate:se,setWeekday:gr,getWeekday:pr,setWeekend:vr};function Nt(t,a,s){let n=!0;for(;n;)n=!1,s.forEach(function(i){const h="^\\s*"+i+"\\s*$",f=new RegExp(h);t[0].match(f)&&(a[i]=!0,t.shift(1),n=!0)})}l(Nt,"getTaskTags");var Ir=l(function(){wt.debug("Something is calling, setConf, remove the call")},"setConf"),ee={monday:Me,tuesday:Ee,wednesday:Se,thursday:Ce,friday:De,saturday:_e,sunday:we},Ar=l((t,a)=>{let s=[...t].map(()=>-1/0),n=[...t].sort((h,f)=>h.startTime-f.startTime||h.order-f.order),i=0;for(const h of n)for(let f=0;f<s.length;f++)if(h.startTime>=s[f]){s[f]=h.endTime,h.order=f+a,f>i&&(i=f);break}return i},"getMaxIntersections"),et,Lr=l(function(t,a,s,n){const i=ct().gantt,h=ct().securityLevel;let f;h==="sandbox"&&(f=gt("#i"+a));const _=h==="sandbox"?gt(f.nodes()[0].contentDocument.body):gt("body"),Y=h==="sandbox"?f.nodes()[0].contentDocument:document,E=Y.getElementById(a);et=E.parentElement.offsetWidth,et===void 0&&(et=1200),i.useWidth!==void 0&&(et=i.useWidth);const p=n.db.getTasks();let M=[];for(const y of p)M.push(y.type);M=U(M);const P={};let V=2*i.topPadding;if(n.db.getDisplayMode()==="compact"||i.displayMode==="compact"){const y={};for(const b of p)y[b.section]===void 0?y[b.section]=[b]:y[b.section].push(b);let T=0;for(const b of Object.keys(y)){const x=Ar(y[b],T)+1;T+=x,V+=x*(i.barHeight+i.barGap),P[b]=x}}else{V+=p.length*(i.barHeight+i.barGap);for(const y of M)P[y]=p.filter(T=>T.type===y).length}E.setAttribute("viewBox","0 0 "+et+" "+V);const B=_.select(`[id="${a}"]`),S=ye().domain([ge(p,function(y){return y.startTime}),pe(p,function(y){return y.endTime})]).rangeRound([0,et-i.leftPadding-i.rightPadding]);function v(y,T){const b=y.startTime,x=T.startTime;let g=0;return b>x?g=1:b<x&&(g=-1),g}l(v,"taskCompare"),p.sort(v),C(p,et,V),ve(B,V,et,i.useMaxWidth),B.append("text").text(n.db.getDiagramTitle()).attr("x",et/2).attr("y",i.titleTopMargin).attr("class","titleText");function C(y,T,b){const x=i.barHeight,g=x+i.barGap,w=i.topPadding,c=i.leftPadding,d=xe().domain([0,M.length]).range(["#00B9FA","#F95002"]).interpolate(Te);F(g,w,c,T,b,y,n.db.getExcludes(),n.db.getIncludes()),G(c,w,T,b),L(y,g,w,c,x,d,T),H(g,w),$(c,w,T,b)}l(C,"makeGantt");function L(y,T,b,x,g,w,c){y.sort((o,e)=>o.vert===e.vert?0:o.vert?1:-1);const k=[...new Set(y.map(o=>o.order))].map(o=>y.find(e=>e.order===o));B.append("g").selectAll("rect").data(k).enter().append("rect").attr("x",0).attr("y",function(o,e){return e=o.order,e*T+b-2}).attr("width",function(){return c-i.rightPadding/2}).attr("height",T).attr("class",function(o){for(const[e,I]of M.entries())if(o.type===I)return"section section"+e%i.numberSectionStyles;return"section section0"}).enter();const u=B.append("g").selectAll("rect").data(y).enter(),m=n.db.getLinks();if(u.append("rect").attr("id",function(o){return o.id}).attr("rx",3).attr("ry",3).attr("x",function(o){return o.milestone?S(o.startTime)+x+.5*(S(o.endTime)-S(o.startTime))-.5*g:S(o.startTime)+x}).attr("y",function(o,e){return e=o.order,o.vert?i.gridLineStartPadding:e*T+b}).attr("width",function(o){return o.milestone?g:o.vert?.08*g:S(o.renderEndTime||o.endTime)-S(o.startTime)}).attr("height",function(o){return o.vert?p.length*(i.barHeight+i.barGap)+i.barHeight*2:g}).attr("transform-origin",function(o,e){return e=o.order,(S(o.startTime)+x+.5*(S(o.endTime)-S(o.startTime))).toString()+"px "+(e*T+b+.5*g).toString()+"px"}).attr("class",function(o){const e="task";let I="";o.classes.length>0&&(I=o.classes.join(" "));let D=0;for(const[N,W]of M.entries())o.type===W&&(D=N%i.numberSectionStyles);let A="";return o.active?o.crit?A+=" activeCrit":A=" active":o.done?o.crit?A=" doneCrit":A=" done":o.crit&&(A+=" crit"),A.length===0&&(A=" task"),o.milestone&&(A=" milestone "+A),o.vert&&(A=" vert "+A),A+=D,A+=" "+I,e+A}),u.append("text").attr("id",function(o){return o.id+"-text"}).text(function(o){return o.task}).attr("font-size",i.fontSize).attr("x",function(o){let e=S(o.startTime),I=S(o.renderEndTime||o.endTime);if(o.milestone&&(e+=.5*(S(o.endTime)-S(o.startTime))-.5*g,I=e+g),o.vert)return S(o.startTime)+x;const D=this.getBBox().width;return D>I-e?I+D+1.5*i.leftPadding>c?e+x-5:I+x+5:(I-e)/2+e+x}).attr("y",function(o,e){return o.vert?i.gridLineStartPadding+p.length*(i.barHeight+i.barGap)+60:(e=o.order,e*T+i.barHeight/2+(i.fontSize/2-2)+b)}).attr("text-height",g).attr("class",function(o){const e=S(o.startTime);let I=S(o.endTime);o.milestone&&(I=e+g);const D=this.getBBox().width;let A="";o.classes.length>0&&(A=o.classes.join(" "));let N=0;for(const[O,Q]of M.entries())o.type===Q&&(N=O%i.numberSectionStyles);let W="";return o.active&&(o.crit?W="activeCritText"+N:W="activeText"+N),o.done?o.crit?W=W+" doneCritText"+N:W=W+" doneText"+N:o.crit&&(W=W+" critText"+N),o.milestone&&(W+=" milestoneText"),o.vert&&(W+=" vertText"),D>I-e?I+D+1.5*i.leftPadding>c?A+" taskTextOutsideLeft taskTextOutside"+N+" "+W:A+" taskTextOutsideRight taskTextOutside"+N+" "+W+" width-"+D:A+" taskText taskText"+N+" "+W+" width-"+D}),ct().securityLevel==="sandbox"){let o;o=gt("#i"+a);const e=o.nodes()[0].contentDocument;u.filter(function(I){return m.has(I.id)}).each(function(I){var D=e.querySelector("#"+I.id),A=e.querySelector("#"+I.id+"-text");const N=D.parentNode;var W=e.createElement("a");W.setAttribute("xlink:href",m.get(I.id)),W.setAttribute("target","_top"),N.appendChild(W),W.appendChild(D),W.appendChild(A)})}}l(L,"drawRects");function F(y,T,b,x,g,w,c,d){if(c.length===0&&d.length===0)return;let k,u;for(const{startTime:D,endTime:A}of w)(k===void 0||D<k)&&(k=D),(u===void 0||A>u)&&(u=A);if(!k||!u)return;if(X(u).diff(X(k),"year")>5){wt.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}const m=n.db.getDateFormat(),r=[];let o=null,e=X(k);for(;e.valueOf()<=u;)n.db.isInvalidDate(e,m,c,d)?o?o.end=e:o={start:e,end:e}:o&&(r.push(o),o=null),e=e.add(1,"d");B.append("g").selectAll("rect").data(r).enter().append("rect").attr("id",D=>"exclude-"+D.start.format("YYYY-MM-DD")).attr("x",D=>S(D.start.startOf("day"))+b).attr("y",i.gridLineStartPadding).attr("width",D=>S(D.end.endOf("day"))-S(D.start.startOf("day"))).attr("height",g-T-i.gridLineStartPadding).attr("transform-origin",function(D,A){return(S(D.start)+b+.5*(S(D.end)-S(D.start))).toString()+"px "+(A*y+.5*g).toString()+"px"}).attr("class","exclude-range")}l(F,"drawExcludeDays");function G(y,T,b,x){const g=n.db.getDateFormat(),w=n.db.getAxisFormat();let c;w?c=w:g==="D"?c="%d":c=i.axisFormat??"%Y-%m-%d";let d=be(S).tickSize(-x+T+i.gridLineStartPadding).tickFormat(qt(c));const u=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(n.db.getTickInterval()||i.tickInterval);if(u!==null){const m=u[1],r=u[2],o=n.db.getWeekday()||i.weekday;switch(r){case"millisecond":d.ticks(Zt.every(m));break;case"second":d.ticks(jt.every(m));break;case"minute":d.ticks(Ut.every(m));break;case"hour":d.ticks(Xt.every(m));break;case"day":d.ticks(Ht.every(m));break;case"week":d.ticks(ee[o].every(m));break;case"month":d.ticks(Gt.every(m));break}}if(B.append("g").attr("class","grid").attr("transform","translate("+y+", "+(x-50)+")").call(d).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),n.db.topAxisEnabled()||i.topAxis){let m=Ie(S).tickSize(-x+T+i.gridLineStartPadding).tickFormat(qt(c));if(u!==null){const r=u[1],o=u[2],e=n.db.getWeekday()||i.weekday;switch(o){case"millisecond":m.ticks(Zt.every(r));break;case"second":m.ticks(jt.every(r));break;case"minute":m.ticks(Ut.every(r));break;case"hour":m.ticks(Xt.every(r));break;case"day":m.ticks(Ht.every(r));break;case"week":m.ticks(ee[e].every(r));break;case"month":m.ticks(Gt.every(r));break}}B.append("g").attr("class","grid").attr("transform","translate("+y+", "+T+")").call(m).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}l(G,"makeGrid");function H(y,T){let b=0;const x=Object.keys(P).map(g=>[g,P[g]]);B.append("g").selectAll("text").data(x).enter().append(function(g){const w=g[0].split(Ae.lineBreakRegex),c=-(w.length-1)/2,d=Y.createElementNS("http://www.w3.org/2000/svg","text");d.setAttribute("dy",c+"em");for(const[k,u]of w.entries()){const m=Y.createElementNS("http://www.w3.org/2000/svg","tspan");m.setAttribute("alignment-baseline","central"),m.setAttribute("x","10"),k>0&&m.setAttribute("dy","1em"),m.textContent=u,d.appendChild(m)}return d}).attr("x",10).attr("y",function(g,w){if(w>0)for(let c=0;c<w;c++)return b+=x[w-1][1],g[1]*y/2+b*y+T;else return g[1]*y/2+T}).attr("font-size",i.sectionFontSize).attr("class",function(g){for(const[w,c]of M.entries())if(g[0]===c)return"sectionTitle sectionTitle"+w%i.numberSectionStyles;return"sectionTitle"})}l(H,"vertLabels");function $(y,T,b,x){const g=n.db.getTodayMarker();if(g==="off")return;const w=B.append("g").attr("class","today"),c=new Date,d=w.append("line");d.attr("x1",S(c)+y).attr("x2",S(c)+y).attr("y1",i.titleTopMargin).attr("y2",x-i.titleTopMargin).attr("class","today"),g!==""&&d.attr("style",g.replace(/,/g,";"))}l($,"drawToday");function U(y){const T={},b=[];for(let x=0,g=y.length;x<g;++x)Object.prototype.hasOwnProperty.call(T,y[x])||(T[y[x]]=!0,b.push(y[x]));return b}l(U,"checkUnique")},"draw"),Fr={setConf:Ir,draw:Lr},Yr=l(t=>`
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

  .vert {
    stroke: ${t.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${t.vertLineColor} !important;
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
`,"getStyles"),Wr=Yr,Br={parser:Ue,db:Mr,renderer:Fr,styles:Wr};export{Br as diagram};

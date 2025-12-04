import{_ as l,g as ut,s as dt,t as ft,q as ht,a as kt,b as mt,c as ce,d as ge,aE as yt,aF as gt,aG as pt,e as vt,R as xt,aH as Tt,aI as X,l as we,aJ as bt,aK as qe,aL as Ge,aM as wt,aN as _t,aO as Dt,aP as Ct,aQ as St,aR as Et,aS as Mt,aT as He,aU as Xe,aV as Ue,aW as je,aX as Ze,aY as It,k as At,j as Lt,z as Ft,u as Yt}from"./mermaid-vendor-BWqoPkHO.js";import{g as Ae}from"./react-vendor-DEwriMA6.js";import"./feature-graph-BnXy9sZT.js";import"./graph-vendor-B-X5JegA.js";import"./ui-vendor-CeCm8EER.js";import"./utils-vendor-BysuhMZA.js";var pe={exports:{}},Wt=pe.exports,$e;function Pt(){return $e||($e=1,function(e,n){(function(r,i){e.exports=i()})(Wt,function(){var r="day";return function(i,a,m){var f=function(M){return M.add(4-M.isoWeekday(),r)},_=a.prototype;_.isoWeekYear=function(){return f(this).year()},_.isoWeek=function(M){if(!this.$utils().u(M))return this.add(7*(M-this.isoWeek()),r);var g,I,V,O,B=f(this),S=(g=this.isoWeekYear(),I=this.$u,V=(I?m.utc:m)().year(g).startOf("year"),O=4-V.isoWeekday(),V.isoWeekday()>4&&(O+=7),V.add(O,r));return B.diff(S,"week")+1},_.isoWeekday=function(M){return this.$utils().u(M)?this.day()||7:this.day(this.day()%7?M:M-7)};var Y=_.startOf;_.startOf=function(M,g){var I=this.$utils(),V=!!I.u(g)||g;return I.p(M)==="isoweek"?V?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):Y.bind(this)(M,g)}}})}(pe)),pe.exports}var Vt=Pt();const Ot=Ae(Vt);var ve={exports:{}},zt=ve.exports,Qe;function Rt(){return Qe||(Qe=1,function(e,n){(function(r,i){e.exports=i()})(zt,function(){var r={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},i=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,a=/\d/,m=/\d\d/,f=/\d\d?/,_=/\d*[^-_:/,()\s\d]+/,Y={},M=function(p){return(p=+p)+(p>68?1900:2e3)},g=function(p){return function(C){this[p]=+C}},I=[/[+-]\d\d:?(\d\d)?|Z/,function(p){(this.zone||(this.zone={})).offset=function(C){if(!C||C==="Z")return 0;var L=C.match(/([+-]|\d\d)/g),F=60*L[1]+(+L[2]||0);return F===0?0:L[0]==="+"?-F:F}(p)}],V=function(p){var C=Y[p];return C&&(C.indexOf?C:C.s.concat(C.f))},O=function(p,C){var L,F=Y.meridiem;if(F){for(var G=1;G<=24;G+=1)if(p.indexOf(F(G,0,C))>-1){L=G>12;break}}else L=p===(C?"pm":"PM");return L},B={A:[_,function(p){this.afternoon=O(p,!1)}],a:[_,function(p){this.afternoon=O(p,!0)}],Q:[a,function(p){this.month=3*(p-1)+1}],S:[a,function(p){this.milliseconds=100*+p}],SS:[m,function(p){this.milliseconds=10*+p}],SSS:[/\d{3}/,function(p){this.milliseconds=+p}],s:[f,g("seconds")],ss:[f,g("seconds")],m:[f,g("minutes")],mm:[f,g("minutes")],H:[f,g("hours")],h:[f,g("hours")],HH:[f,g("hours")],hh:[f,g("hours")],D:[f,g("day")],DD:[m,g("day")],Do:[_,function(p){var C=Y.ordinal,L=p.match(/\d+/);if(this.day=L[0],C)for(var F=1;F<=31;F+=1)C(F).replace(/\[|\]/g,"")===p&&(this.day=F)}],w:[f,g("week")],ww:[m,g("week")],M:[f,g("month")],MM:[m,g("month")],MMM:[_,function(p){var C=V("months"),L=(V("monthsShort")||C.map(function(F){return F.slice(0,3)})).indexOf(p)+1;if(L<1)throw new Error;this.month=L%12||L}],MMMM:[_,function(p){var C=V("months").indexOf(p)+1;if(C<1)throw new Error;this.month=C%12||C}],Y:[/[+-]?\d+/,g("year")],YY:[m,function(p){this.year=M(p)}],YYYY:[/\d{4}/,g("year")],Z:I,ZZ:I};function S(p){var C,L;C=p,L=Y&&Y.formats;for(var F=(p=C.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,function(b,x,k){var w=k&&k.toUpperCase();return x||L[k]||r[k]||L[w].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,function(c,u,h){return u||h.slice(1)})})).match(i),G=F.length,H=0;H<G;H+=1){var $=F[H],U=B[$],y=U&&U[0],T=U&&U[1];F[H]=T?{regex:y,parser:T}:$.replace(/^\[|\]$/g,"")}return function(b){for(var x={},k=0,w=0;k<G;k+=1){var c=F[k];if(typeof c=="string")w+=c.length;else{var u=c.regex,h=c.parser,d=b.slice(w),v=u.exec(d)[0];h.call(x,v),b=b.replace(v,"")}}return function(s){var o=s.afternoon;if(o!==void 0){var t=s.hours;o?t<12&&(s.hours+=12):t===12&&(s.hours=0),delete s.afternoon}}(x),x}}return function(p,C,L){L.p.customParseFormat=!0,p&&p.parseTwoDigitYear&&(M=p.parseTwoDigitYear);var F=C.prototype,G=F.parse;F.parse=function(H){var $=H.date,U=H.utc,y=H.args;this.$u=U;var T=y[1];if(typeof T=="string"){var b=y[2]===!0,x=y[3]===!0,k=b||x,w=y[2];x&&(w=y[2]),Y=this.$locale(),!b&&w&&(Y=L.Ls[w]),this.$d=function(d,v,s,o){try{if(["x","X"].indexOf(v)>-1)return new Date((v==="X"?1e3:1)*d);var t=S(v)(d),A=t.year,D=t.month,E=t.day,N=t.hours,W=t.minutes,P=t.seconds,Q=t.milliseconds,ae=t.zone,ie=t.week,de=new Date,fe=E||(A||D?1:de.getDate()),oe=A||de.getFullYear(),z=0;A&&!D||(z=D>0?D-1:de.getMonth());var Z,q=N||0,se=W||0,K=P||0,re=Q||0;return ae?new Date(Date.UTC(oe,z,fe,q,se,K,re+60*ae.offset*1e3)):s?new Date(Date.UTC(oe,z,fe,q,se,K,re)):(Z=new Date(oe,z,fe,q,se,K,re),ie&&(Z=o(Z).week(ie).toDate()),Z)}catch{return new Date("")}}($,T,U,L),this.init(),w&&w!==!0&&(this.$L=this.locale(w).$L),k&&$!=this.format(T)&&(this.$d=new Date("")),Y={}}else if(T instanceof Array)for(var c=T.length,u=1;u<=c;u+=1){y[1]=T[u-1];var h=L.apply(this,y);if(h.isValid()){this.$d=h.$d,this.$L=h.$L,this.init();break}u===c&&(this.$d=new Date(""))}else G.call(this,H)}}})}(ve)),ve.exports}var Nt=Rt();const Bt=Ae(Nt);var xe={exports:{}},qt=xe.exports,Ke;function Gt(){return Ke||(Ke=1,function(e,n){(function(r,i){e.exports=i()})(qt,function(){return function(r,i){var a=i.prototype,m=a.format;a.format=function(f){var _=this,Y=this.$locale();if(!this.isValid())return m.bind(this)(f);var M=this.$utils(),g=(f||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,function(I){switch(I){case"Q":return Math.ceil((_.$M+1)/3);case"Do":return Y.ordinal(_.$D);case"gggg":return _.weekYear();case"GGGG":return _.isoWeekYear();case"wo":return Y.ordinal(_.week(),"W");case"w":case"ww":return M.s(_.week(),I==="w"?1:2,"0");case"W":case"WW":return M.s(_.isoWeek(),I==="W"?1:2,"0");case"k":case"kk":return M.s(String(_.$H===0?24:_.$H),I==="k"?1:2,"0");case"X":return Math.floor(_.$d.getTime()/1e3);case"x":return _.$d.getTime();case"z":return"["+_.offsetName()+"]";case"zzz":return"["+_.offsetName("long")+"]";default:return I}});return m.bind(this)(g)}}})}(xe)),xe.exports}var Ht=Gt();const Xt=Ae(Ht);var Se=function(){var e=l(function(w,c,u,h){for(u=u||{},h=w.length;h--;u[w[h]]=c);return u},"o"),n=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],r=[1,26],i=[1,27],a=[1,28],m=[1,29],f=[1,30],_=[1,31],Y=[1,32],M=[1,33],g=[1,34],I=[1,9],V=[1,10],O=[1,11],B=[1,12],S=[1,13],p=[1,14],C=[1,15],L=[1,16],F=[1,19],G=[1,20],H=[1,21],$=[1,22],U=[1,23],y=[1,25],T=[1,35],b={trace:l(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:l(function(c,u,h,d,v,s,o){var t=s.length-1;switch(v){case 1:return s[t-1];case 2:this.$=[];break;case 3:s[t-1].push(s[t]),this.$=s[t-1];break;case 4:case 5:this.$=s[t];break;case 6:case 7:this.$=[];break;case 8:d.setWeekday("monday");break;case 9:d.setWeekday("tuesday");break;case 10:d.setWeekday("wednesday");break;case 11:d.setWeekday("thursday");break;case 12:d.setWeekday("friday");break;case 13:d.setWeekday("saturday");break;case 14:d.setWeekday("sunday");break;case 15:d.setWeekend("friday");break;case 16:d.setWeekend("saturday");break;case 17:d.setDateFormat(s[t].substr(11)),this.$=s[t].substr(11);break;case 18:d.enableInclusiveEndDates(),this.$=s[t].substr(18);break;case 19:d.TopAxis(),this.$=s[t].substr(8);break;case 20:d.setAxisFormat(s[t].substr(11)),this.$=s[t].substr(11);break;case 21:d.setTickInterval(s[t].substr(13)),this.$=s[t].substr(13);break;case 22:d.setExcludes(s[t].substr(9)),this.$=s[t].substr(9);break;case 23:d.setIncludes(s[t].substr(9)),this.$=s[t].substr(9);break;case 24:d.setTodayMarker(s[t].substr(12)),this.$=s[t].substr(12);break;case 27:d.setDiagramTitle(s[t].substr(6)),this.$=s[t].substr(6);break;case 28:this.$=s[t].trim(),d.setAccTitle(this.$);break;case 29:case 30:this.$=s[t].trim(),d.setAccDescription(this.$);break;case 31:d.addSection(s[t].substr(8)),this.$=s[t].substr(8);break;case 33:d.addTask(s[t-1],s[t]),this.$="task";break;case 34:this.$=s[t-1],d.setClickEvent(s[t-1],s[t],null);break;case 35:this.$=s[t-2],d.setClickEvent(s[t-2],s[t-1],s[t]);break;case 36:this.$=s[t-2],d.setClickEvent(s[t-2],s[t-1],null),d.setLink(s[t-2],s[t]);break;case 37:this.$=s[t-3],d.setClickEvent(s[t-3],s[t-2],s[t-1]),d.setLink(s[t-3],s[t]);break;case 38:this.$=s[t-2],d.setClickEvent(s[t-2],s[t],null),d.setLink(s[t-2],s[t-1]);break;case 39:this.$=s[t-3],d.setClickEvent(s[t-3],s[t-1],s[t]),d.setLink(s[t-3],s[t-2]);break;case 40:this.$=s[t-1],d.setLink(s[t-1],s[t]);break;case 41:case 47:this.$=s[t-1]+" "+s[t];break;case 42:case 43:case 45:this.$=s[t-2]+" "+s[t-1]+" "+s[t];break;case 44:case 46:this.$=s[t-3]+" "+s[t-2]+" "+s[t-1]+" "+s[t];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},e(n,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:r,13:i,14:a,15:m,16:f,17:_,18:Y,19:18,20:M,21:g,22:I,23:V,24:O,25:B,26:S,27:p,28:C,29:L,30:F,31:G,33:H,35:$,36:U,37:24,38:y,40:T},e(n,[2,7],{1:[2,1]}),e(n,[2,3]),{9:36,11:17,12:r,13:i,14:a,15:m,16:f,17:_,18:Y,19:18,20:M,21:g,22:I,23:V,24:O,25:B,26:S,27:p,28:C,29:L,30:F,31:G,33:H,35:$,36:U,37:24,38:y,40:T},e(n,[2,5]),e(n,[2,6]),e(n,[2,17]),e(n,[2,18]),e(n,[2,19]),e(n,[2,20]),e(n,[2,21]),e(n,[2,22]),e(n,[2,23]),e(n,[2,24]),e(n,[2,25]),e(n,[2,26]),e(n,[2,27]),{32:[1,37]},{34:[1,38]},e(n,[2,30]),e(n,[2,31]),e(n,[2,32]),{39:[1,39]},e(n,[2,8]),e(n,[2,9]),e(n,[2,10]),e(n,[2,11]),e(n,[2,12]),e(n,[2,13]),e(n,[2,14]),e(n,[2,15]),e(n,[2,16]),{41:[1,40],43:[1,41]},e(n,[2,4]),e(n,[2,28]),e(n,[2,29]),e(n,[2,33]),e(n,[2,34],{42:[1,42],43:[1,43]}),e(n,[2,40],{41:[1,44]}),e(n,[2,35],{43:[1,45]}),e(n,[2,36]),e(n,[2,38],{42:[1,46]}),e(n,[2,37]),e(n,[2,39])],defaultActions:{},parseError:l(function(c,u){if(u.recoverable)this.trace(c);else{var h=new Error(c);throw h.hash=u,h}},"parseError"),parse:l(function(c){var u=this,h=[0],d=[],v=[null],s=[],o=this.table,t="",A=0,D=0,E=2,N=1,W=s.slice.call(arguments,1),P=Object.create(this.lexer),Q={yy:{}};for(var ae in this.yy)Object.prototype.hasOwnProperty.call(this.yy,ae)&&(Q.yy[ae]=this.yy[ae]);P.setInput(c,Q.yy),Q.yy.lexer=P,Q.yy.parser=this,typeof P.yylloc>"u"&&(P.yylloc={});var ie=P.yylloc;s.push(ie);var de=P.options&&P.options.ranges;typeof Q.yy.parseError=="function"?this.parseError=Q.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function fe(j){h.length=h.length-2*j,v.length=v.length-j,s.length=s.length-j}l(fe,"popStack");function oe(){var j;return j=d.pop()||P.lex()||N,typeof j!="number"&&(j instanceof Array&&(d=j,j=d.pop()),j=u.symbols_[j]||j),j}l(oe,"lex");for(var z,Z,q,se,K={},re,J,Be,ye;;){if(Z=h[h.length-1],this.defaultActions[Z]?q=this.defaultActions[Z]:((z===null||typeof z>"u")&&(z=oe()),q=o[Z]&&o[Z][z]),typeof q>"u"||!q.length||!q[0]){var Ce="";ye=[];for(re in o[Z])this.terminals_[re]&&re>E&&ye.push("'"+this.terminals_[re]+"'");P.showPosition?Ce="Parse error on line "+(A+1)+`:
`+P.showPosition()+`
Expecting `+ye.join(", ")+", got '"+(this.terminals_[z]||z)+"'":Ce="Parse error on line "+(A+1)+": Unexpected "+(z==N?"end of input":"'"+(this.terminals_[z]||z)+"'"),this.parseError(Ce,{text:P.match,token:this.terminals_[z]||z,line:P.yylineno,loc:ie,expected:ye})}if(q[0]instanceof Array&&q.length>1)throw new Error("Parse Error: multiple actions possible at state: "+Z+", token: "+z);switch(q[0]){case 1:h.push(z),v.push(P.yytext),s.push(P.yylloc),h.push(q[1]),z=null,D=P.yyleng,t=P.yytext,A=P.yylineno,ie=P.yylloc;break;case 2:if(J=this.productions_[q[1]][1],K.$=v[v.length-J],K._$={first_line:s[s.length-(J||1)].first_line,last_line:s[s.length-1].last_line,first_column:s[s.length-(J||1)].first_column,last_column:s[s.length-1].last_column},de&&(K._$.range=[s[s.length-(J||1)].range[0],s[s.length-1].range[1]]),se=this.performAction.apply(K,[t,D,A,Q.yy,q[1],v,s].concat(W)),typeof se<"u")return se;J&&(h=h.slice(0,-1*J*2),v=v.slice(0,-1*J),s=s.slice(0,-1*J)),h.push(this.productions_[q[1]][0]),v.push(K.$),s.push(K._$),Be=o[h[h.length-2]][h[h.length-1]],h.push(Be);break;case 3:return!0}}return!0},"parse")},x=function(){var w={EOF:1,parseError:l(function(u,h){if(this.yy.parser)this.yy.parser.parseError(u,h);else throw new Error(u)},"parseError"),setInput:l(function(c,u){return this.yy=u||this.yy||{},this._input=c,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:l(function(){var c=this._input[0];this.yytext+=c,this.yyleng++,this.offset++,this.match+=c,this.matched+=c;var u=c.match(/(?:\r\n?|\n).*/g);return u?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),c},"input"),unput:l(function(c){var u=c.length,h=c.split(/(?:\r\n?|\n)/g);this._input=c+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-u),this.offset-=u;var d=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),h.length-1&&(this.yylineno-=h.length-1);var v=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:h?(h.length===d.length?this.yylloc.first_column:0)+d[d.length-h.length].length-h[0].length:this.yylloc.first_column-u},this.options.ranges&&(this.yylloc.range=[v[0],v[0]+this.yyleng-u]),this.yyleng=this.yytext.length,this},"unput"),more:l(function(){return this._more=!0,this},"more"),reject:l(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:l(function(c){this.unput(this.match.slice(c))},"less"),pastInput:l(function(){var c=this.matched.substr(0,this.matched.length-this.match.length);return(c.length>20?"...":"")+c.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:l(function(){var c=this.match;return c.length<20&&(c+=this._input.substr(0,20-c.length)),(c.substr(0,20)+(c.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:l(function(){var c=this.pastInput(),u=new Array(c.length+1).join("-");return c+this.upcomingInput()+`
`+u+"^"},"showPosition"),test_match:l(function(c,u){var h,d,v;if(this.options.backtrack_lexer&&(v={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(v.yylloc.range=this.yylloc.range.slice(0))),d=c[0].match(/(?:\r\n?|\n).*/g),d&&(this.yylineno+=d.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:d?d[d.length-1].length-d[d.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+c[0].length},this.yytext+=c[0],this.match+=c[0],this.matches=c,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(c[0].length),this.matched+=c[0],h=this.performAction.call(this,this.yy,this,u,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),h)return h;if(this._backtrack){for(var s in v)this[s]=v[s];return!1}return!1},"test_match"),next:l(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var c,u,h,d;this._more||(this.yytext="",this.match="");for(var v=this._currentRules(),s=0;s<v.length;s++)if(h=this._input.match(this.rules[v[s]]),h&&(!u||h[0].length>u[0].length)){if(u=h,d=s,this.options.backtrack_lexer){if(c=this.test_match(h,v[s]),c!==!1)return c;if(this._backtrack){u=!1;continue}else return!1}else if(!this.options.flex)break}return u?(c=this.test_match(u,v[d]),c!==!1?c:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:l(function(){var u=this.next();return u||this.lex()},"lex"),begin:l(function(u){this.conditionStack.push(u)},"begin"),popState:l(function(){var u=this.conditionStack.length-1;return u>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:l(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:l(function(u){return u=this.conditionStack.length-1-Math.abs(u||0),u>=0?this.conditionStack[u]:"INITIAL"},"topState"),pushState:l(function(u){this.begin(u)},"pushState"),stateStackSize:l(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:l(function(u,h,d,v){switch(d){case 0:return this.begin("open_directive"),"open_directive";case 1:return this.begin("acc_title"),31;case 2:return this.popState(),"acc_title_value";case 3:return this.begin("acc_descr"),33;case 4:return this.popState(),"acc_descr_value";case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}};return w}();b.lexer=x;function k(){this.yy={}}return l(k,"Parser"),k.prototype=b,b.Parser=k,new k}();Se.parser=Se;var Ut=Se;X.extend(Ot);X.extend(Bt);X.extend(Xt);var Je={friday:5,saturday:6},ee="",Le="",Fe=void 0,Ye="",he=[],ke=[],We=new Map,Pe=[],_e=[],ue="",Ve="",rt=["active","done","crit","milestone","vert"],Oe=[],me=!1,ze=!1,Re="sunday",De="saturday",Ee=0,jt=l(function(){Pe=[],_e=[],ue="",Oe=[],Te=0,Ie=void 0,be=void 0,R=[],ee="",Le="",Ve="",Fe=void 0,Ye="",he=[],ke=[],me=!1,ze=!1,Ee=0,We=new Map,Ft(),Re="sunday",De="saturday"},"clear"),Zt=l(function(e){Le=e},"setAxisFormat"),$t=l(function(){return Le},"getAxisFormat"),Qt=l(function(e){Fe=e},"setTickInterval"),Kt=l(function(){return Fe},"getTickInterval"),Jt=l(function(e){Ye=e},"setTodayMarker"),er=l(function(){return Ye},"getTodayMarker"),tr=l(function(e){ee=e},"setDateFormat"),rr=l(function(){me=!0},"enableInclusiveEndDates"),sr=l(function(){return me},"endDatesAreInclusive"),nr=l(function(){ze=!0},"enableTopAxis"),ar=l(function(){return ze},"topAxisEnabled"),ir=l(function(e){Ve=e},"setDisplayMode"),or=l(function(){return Ve},"getDisplayMode"),cr=l(function(){return ee},"getDateFormat"),lr=l(function(e){he=e.toLowerCase().split(/[\s,]+/)},"setIncludes"),ur=l(function(){return he},"getIncludes"),dr=l(function(e){ke=e.toLowerCase().split(/[\s,]+/)},"setExcludes"),fr=l(function(){return ke},"getExcludes"),hr=l(function(){return We},"getLinks"),kr=l(function(e){ue=e,Pe.push(e)},"addSection"),mr=l(function(){return Pe},"getSections"),yr=l(function(){let e=et();const n=10;let r=0;for(;!e&&r<n;)e=et(),r++;return _e=R,_e},"getTasks"),st=l(function(e,n,r,i){return i.includes(e.format(n.trim()))?!1:r.includes("weekends")&&(e.isoWeekday()===Je[De]||e.isoWeekday()===Je[De]+1)||r.includes(e.format("dddd").toLowerCase())?!0:r.includes(e.format(n.trim()))},"isInvalidDate"),gr=l(function(e){Re=e},"setWeekday"),pr=l(function(){return Re},"getWeekday"),vr=l(function(e){De=e},"setWeekend"),nt=l(function(e,n,r,i){if(!r.length||e.manualEndTime)return;let a;e.startTime instanceof Date?a=X(e.startTime):a=X(e.startTime,n,!0),a=a.add(1,"d");let m;e.endTime instanceof Date?m=X(e.endTime):m=X(e.endTime,n,!0);const[f,_]=xr(a,m,n,r,i);e.endTime=f.toDate(),e.renderEndTime=_},"checkTaskDates"),xr=l(function(e,n,r,i,a){let m=!1,f=null;for(;e<=n;)m||(f=n.toDate()),m=st(e,r,i,a),m&&(n=n.add(1,"d")),e=e.add(1,"d");return[n,f]},"fixTaskDates"),Me=l(function(e,n,r){r=r.trim();const a=/^after\s+(?<ids>[\d\w- ]+)/.exec(r);if(a!==null){let f=null;for(const Y of a.groups.ids.split(" ")){let M=ne(Y);M!==void 0&&(!f||M.endTime>f.endTime)&&(f=M)}if(f)return f.endTime;const _=new Date;return _.setHours(0,0,0,0),_}let m=X(r,n.trim(),!0);if(m.isValid())return m.toDate();{we.debug("Invalid date:"+r),we.debug("With date format:"+n.trim());const f=new Date(r);if(f===void 0||isNaN(f.getTime())||f.getFullYear()<-1e4||f.getFullYear()>1e4)throw new Error("Invalid date:"+r);return f}},"getStartDate"),at=l(function(e){const n=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(e.trim());return n!==null?[Number.parseFloat(n[1]),n[2]]:[NaN,"ms"]},"parseDuration"),it=l(function(e,n,r,i=!1){r=r.trim();const m=/^until\s+(?<ids>[\d\w- ]+)/.exec(r);if(m!==null){let g=null;for(const V of m.groups.ids.split(" ")){let O=ne(V);O!==void 0&&(!g||O.startTime<g.startTime)&&(g=O)}if(g)return g.startTime;const I=new Date;return I.setHours(0,0,0,0),I}let f=X(r,n.trim(),!0);if(f.isValid())return i&&(f=f.add(1,"d")),f.toDate();let _=X(e);const[Y,M]=at(r);if(!Number.isNaN(Y)){const g=_.add(Y,M);g.isValid()&&(_=g)}return _.toDate()},"getEndDate"),Te=0,le=l(function(e){return e===void 0?(Te=Te+1,"task"+Te):e},"parseId"),Tr=l(function(e,n){let r;n.substr(0,1)===":"?r=n.substr(1,n.length):r=n;const i=r.split(","),a={};Ne(i,a,rt);for(let f=0;f<i.length;f++)i[f]=i[f].trim();let m="";switch(i.length){case 1:a.id=le(),a.startTime=e.endTime,m=i[0];break;case 2:a.id=le(),a.startTime=Me(void 0,ee,i[0]),m=i[1];break;case 3:a.id=le(i[0]),a.startTime=Me(void 0,ee,i[1]),m=i[2];break}return m&&(a.endTime=it(a.startTime,ee,m,me),a.manualEndTime=X(m,"YYYY-MM-DD",!0).isValid(),nt(a,ee,ke,he)),a},"compileData"),br=l(function(e,n){let r;n.substr(0,1)===":"?r=n.substr(1,n.length):r=n;const i=r.split(","),a={};Ne(i,a,rt);for(let m=0;m<i.length;m++)i[m]=i[m].trim();switch(i.length){case 1:a.id=le(),a.startTime={type:"prevTaskEnd",id:e},a.endTime={data:i[0]};break;case 2:a.id=le(),a.startTime={type:"getStartDate",startData:i[0]},a.endTime={data:i[1]};break;case 3:a.id=le(i[0]),a.startTime={type:"getStartDate",startData:i[1]},a.endTime={data:i[2]};break}return a},"parseData"),Ie,be,R=[],ot={},wr=l(function(e,n){const r={section:ue,type:ue,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:n},task:e,classes:[]},i=br(be,n);r.raw.startTime=i.startTime,r.raw.endTime=i.endTime,r.id=i.id,r.prevTaskId=be,r.active=i.active,r.done=i.done,r.crit=i.crit,r.milestone=i.milestone,r.vert=i.vert,r.order=Ee,Ee++;const a=R.push(r);be=r.id,ot[r.id]=a-1},"addTask"),ne=l(function(e){const n=ot[e];return R[n]},"findTaskById"),_r=l(function(e,n){const r={section:ue,type:ue,description:e,task:e,classes:[]},i=Tr(Ie,n);r.startTime=i.startTime,r.endTime=i.endTime,r.id=i.id,r.active=i.active,r.done=i.done,r.crit=i.crit,r.milestone=i.milestone,r.vert=i.vert,Ie=r,_e.push(r)},"addTaskOrg"),et=l(function(){const e=l(function(r){const i=R[r];let a="";switch(R[r].raw.startTime.type){case"prevTaskEnd":{const m=ne(i.prevTaskId);i.startTime=m.endTime;break}case"getStartDate":a=Me(void 0,ee,R[r].raw.startTime.startData),a&&(R[r].startTime=a);break}return R[r].startTime&&(R[r].endTime=it(R[r].startTime,ee,R[r].raw.endTime.data,me),R[r].endTime&&(R[r].processed=!0,R[r].manualEndTime=X(R[r].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),nt(R[r],ee,ke,he))),R[r].processed},"compileTask");let n=!0;for(const[r,i]of R.entries())e(r),n=n&&i.processed;return n},"compileTasks"),Dr=l(function(e,n){let r=n;ce().securityLevel!=="loose"&&(r=Lt.sanitizeUrl(n)),e.split(",").forEach(function(i){ne(i)!==void 0&&(lt(i,()=>{window.open(r,"_self")}),We.set(i,r))}),ct(e,"clickable")},"setLink"),ct=l(function(e,n){e.split(",").forEach(function(r){let i=ne(r);i!==void 0&&i.classes.push(n)})},"setClass"),Cr=l(function(e,n,r){if(ce().securityLevel!=="loose"||n===void 0)return;let i=[];if(typeof r=="string"){i=r.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let m=0;m<i.length;m++){let f=i[m].trim();f.startsWith('"')&&f.endsWith('"')&&(f=f.substr(1,f.length-2)),i[m]=f}}i.length===0&&i.push(e),ne(e)!==void 0&&lt(e,()=>{Yt.runFunc(n,...i)})},"setClickFun"),lt=l(function(e,n){Oe.push(function(){const r=document.querySelector(`[id="${e}"]`);r!==null&&r.addEventListener("click",function(){n()})},function(){const r=document.querySelector(`[id="${e}-text"]`);r!==null&&r.addEventListener("click",function(){n()})})},"pushFun"),Sr=l(function(e,n,r){e.split(",").forEach(function(i){Cr(i,n,r)}),ct(e,"clickable")},"setClickEvent"),Er=l(function(e){Oe.forEach(function(n){n(e)})},"bindFunctions"),Mr={getConfig:l(()=>ce().gantt,"getConfig"),clear:jt,setDateFormat:tr,getDateFormat:cr,enableInclusiveEndDates:rr,endDatesAreInclusive:sr,enableTopAxis:nr,topAxisEnabled:ar,setAxisFormat:Zt,getAxisFormat:$t,setTickInterval:Qt,getTickInterval:Kt,setTodayMarker:Jt,getTodayMarker:er,setAccTitle:mt,getAccTitle:kt,setDiagramTitle:ht,getDiagramTitle:ft,setDisplayMode:ir,getDisplayMode:or,setAccDescription:dt,getAccDescription:ut,addSection:kr,getSections:mr,getTasks:yr,addTask:wr,findTaskById:ne,addTaskOrg:_r,setIncludes:lr,getIncludes:ur,setExcludes:dr,getExcludes:fr,setClickEvent:Sr,setLink:Dr,getLinks:hr,bindFunctions:Er,parseDuration:at,isInvalidDate:st,setWeekday:gr,getWeekday:pr,setWeekend:vr};function Ne(e,n,r){let i=!0;for(;i;)i=!1,r.forEach(function(a){const m="^\\s*"+a+"\\s*$",f=new RegExp(m);e[0].match(f)&&(n[a]=!0,e.shift(1),i=!0)})}l(Ne,"getTaskTags");var Ir=l(function(){we.debug("Something is calling, setConf, remove the call")},"setConf"),tt={monday:Mt,tuesday:Et,wednesday:St,thursday:Ct,friday:Dt,saturday:_t,sunday:wt},Ar=l((e,n)=>{let r=[...e].map(()=>-1/0),i=[...e].sort((m,f)=>m.startTime-f.startTime||m.order-f.order),a=0;for(const m of i)for(let f=0;f<r.length;f++)if(m.startTime>=r[f]){r[f]=m.endTime,m.order=f+n,f>a&&(a=f);break}return a},"getMaxIntersections"),te,Lr=l(function(e,n,r,i){const a=ce().gantt,m=ce().securityLevel;let f;m==="sandbox"&&(f=ge("#i"+n));const _=m==="sandbox"?ge(f.nodes()[0].contentDocument.body):ge("body"),Y=m==="sandbox"?f.nodes()[0].contentDocument:document,M=Y.getElementById(n);te=M.parentElement.offsetWidth,te===void 0&&(te=1200),a.useWidth!==void 0&&(te=a.useWidth);const g=i.db.getTasks();let I=[];for(const y of g)I.push(y.type);I=U(I);const V={};let O=2*a.topPadding;if(i.db.getDisplayMode()==="compact"||a.displayMode==="compact"){const y={};for(const b of g)y[b.section]===void 0?y[b.section]=[b]:y[b.section].push(b);let T=0;for(const b of Object.keys(y)){const x=Ar(y[b],T)+1;T+=x,O+=x*(a.barHeight+a.barGap),V[b]=x}}else{O+=g.length*(a.barHeight+a.barGap);for(const y of I)V[y]=g.filter(T=>T.type===y).length}M.setAttribute("viewBox","0 0 "+te+" "+O);const B=_.select(`[id="${n}"]`),S=yt().domain([gt(g,function(y){return y.startTime}),pt(g,function(y){return y.endTime})]).rangeRound([0,te-a.leftPadding-a.rightPadding]);function p(y,T){const b=y.startTime,x=T.startTime;let k=0;return b>x?k=1:b<x&&(k=-1),k}l(p,"taskCompare"),g.sort(p),C(g,te,O),vt(B,O,te,a.useMaxWidth),B.append("text").text(i.db.getDiagramTitle()).attr("x",te/2).attr("y",a.titleTopMargin).attr("class","titleText");function C(y,T,b){const x=a.barHeight,k=x+a.barGap,w=a.topPadding,c=a.leftPadding,u=xt().domain([0,I.length]).range(["#00B9FA","#F95002"]).interpolate(Tt);F(k,w,c,T,b,y,i.db.getExcludes(),i.db.getIncludes()),G(c,w,T,b),L(y,k,w,c,x,u,T),H(k,w),$(c,w,T,b)}l(C,"makeGantt");function L(y,T,b,x,k,w,c){y.sort((o,t)=>o.vert===t.vert?0:o.vert?1:-1);const h=[...new Set(y.map(o=>o.order))].map(o=>y.find(t=>t.order===o));B.append("g").selectAll("rect").data(h).enter().append("rect").attr("x",0).attr("y",function(o,t){return t=o.order,t*T+b-2}).attr("width",function(){return c-a.rightPadding/2}).attr("height",T).attr("class",function(o){for(const[t,A]of I.entries())if(o.type===A)return"section section"+t%a.numberSectionStyles;return"section section0"}).enter();const d=B.append("g").selectAll("rect").data(y).enter(),v=i.db.getLinks();if(d.append("rect").attr("id",function(o){return o.id}).attr("rx",3).attr("ry",3).attr("x",function(o){return o.milestone?S(o.startTime)+x+.5*(S(o.endTime)-S(o.startTime))-.5*k:S(o.startTime)+x}).attr("y",function(o,t){return t=o.order,o.vert?a.gridLineStartPadding:t*T+b}).attr("width",function(o){return o.milestone?k:o.vert?.08*k:S(o.renderEndTime||o.endTime)-S(o.startTime)}).attr("height",function(o){return o.vert?g.length*(a.barHeight+a.barGap)+a.barHeight*2:k}).attr("transform-origin",function(o,t){return t=o.order,(S(o.startTime)+x+.5*(S(o.endTime)-S(o.startTime))).toString()+"px "+(t*T+b+.5*k).toString()+"px"}).attr("class",function(o){const t="task";let A="";o.classes.length>0&&(A=o.classes.join(" "));let D=0;for(const[N,W]of I.entries())o.type===W&&(D=N%a.numberSectionStyles);let E="";return o.active?o.crit?E+=" activeCrit":E=" active":o.done?o.crit?E=" doneCrit":E=" done":o.crit&&(E+=" crit"),E.length===0&&(E=" task"),o.milestone&&(E=" milestone "+E),o.vert&&(E=" vert "+E),E+=D,E+=" "+A,t+E}),d.append("text").attr("id",function(o){return o.id+"-text"}).text(function(o){return o.task}).attr("font-size",a.fontSize).attr("x",function(o){let t=S(o.startTime),A=S(o.renderEndTime||o.endTime);if(o.milestone&&(t+=.5*(S(o.endTime)-S(o.startTime))-.5*k,A=t+k),o.vert)return S(o.startTime)+x;const D=this.getBBox().width;return D>A-t?A+D+1.5*a.leftPadding>c?t+x-5:A+x+5:(A-t)/2+t+x}).attr("y",function(o,t){return o.vert?a.gridLineStartPadding+g.length*(a.barHeight+a.barGap)+60:(t=o.order,t*T+a.barHeight/2+(a.fontSize/2-2)+b)}).attr("text-height",k).attr("class",function(o){const t=S(o.startTime);let A=S(o.endTime);o.milestone&&(A=t+k);const D=this.getBBox().width;let E="";o.classes.length>0&&(E=o.classes.join(" "));let N=0;for(const[P,Q]of I.entries())o.type===Q&&(N=P%a.numberSectionStyles);let W="";return o.active&&(o.crit?W="activeCritText"+N:W="activeText"+N),o.done?o.crit?W=W+" doneCritText"+N:W=W+" doneText"+N:o.crit&&(W=W+" critText"+N),o.milestone&&(W+=" milestoneText"),o.vert&&(W+=" vertText"),D>A-t?A+D+1.5*a.leftPadding>c?E+" taskTextOutsideLeft taskTextOutside"+N+" "+W:E+" taskTextOutsideRight taskTextOutside"+N+" "+W+" width-"+D:E+" taskText taskText"+N+" "+W+" width-"+D}),ce().securityLevel==="sandbox"){let o;o=ge("#i"+n);const t=o.nodes()[0].contentDocument;d.filter(function(A){return v.has(A.id)}).each(function(A){var D=t.querySelector("#"+A.id),E=t.querySelector("#"+A.id+"-text");const N=D.parentNode;var W=t.createElement("a");W.setAttribute("xlink:href",v.get(A.id)),W.setAttribute("target","_top"),N.appendChild(W),W.appendChild(D),W.appendChild(E)})}}l(L,"drawRects");function F(y,T,b,x,k,w,c,u){if(c.length===0&&u.length===0)return;let h,d;for(const{startTime:D,endTime:E}of w)(h===void 0||D<h)&&(h=D),(d===void 0||E>d)&&(d=E);if(!h||!d)return;if(X(d).diff(X(h),"year")>5){we.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}const v=i.db.getDateFormat(),s=[];let o=null,t=X(h);for(;t.valueOf()<=d;)i.db.isInvalidDate(t,v,c,u)?o?o.end=t:o={start:t,end:t}:o&&(s.push(o),o=null),t=t.add(1,"d");B.append("g").selectAll("rect").data(s).enter().append("rect").attr("id",function(D){return"exclude-"+D.start.format("YYYY-MM-DD")}).attr("x",function(D){return S(D.start)+b}).attr("y",a.gridLineStartPadding).attr("width",function(D){const E=D.end.add(1,"day");return S(E)-S(D.start)}).attr("height",k-T-a.gridLineStartPadding).attr("transform-origin",function(D,E){return(S(D.start)+b+.5*(S(D.end)-S(D.start))).toString()+"px "+(E*y+.5*k).toString()+"px"}).attr("class","exclude-range")}l(F,"drawExcludeDays");function G(y,T,b,x){let k=bt(S).tickSize(-x+T+a.gridLineStartPadding).tickFormat(qe(i.db.getAxisFormat()||a.axisFormat||"%Y-%m-%d"));const c=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(i.db.getTickInterval()||a.tickInterval);if(c!==null){const u=c[1],h=c[2],d=i.db.getWeekday()||a.weekday;switch(h){case"millisecond":k.ticks(Ze.every(u));break;case"second":k.ticks(je.every(u));break;case"minute":k.ticks(Ue.every(u));break;case"hour":k.ticks(Xe.every(u));break;case"day":k.ticks(He.every(u));break;case"week":k.ticks(tt[d].every(u));break;case"month":k.ticks(Ge.every(u));break}}if(B.append("g").attr("class","grid").attr("transform","translate("+y+", "+(x-50)+")").call(k).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),i.db.topAxisEnabled()||a.topAxis){let u=It(S).tickSize(-x+T+a.gridLineStartPadding).tickFormat(qe(i.db.getAxisFormat()||a.axisFormat||"%Y-%m-%d"));if(c!==null){const h=c[1],d=c[2],v=i.db.getWeekday()||a.weekday;switch(d){case"millisecond":u.ticks(Ze.every(h));break;case"second":u.ticks(je.every(h));break;case"minute":u.ticks(Ue.every(h));break;case"hour":u.ticks(Xe.every(h));break;case"day":u.ticks(He.every(h));break;case"week":u.ticks(tt[v].every(h));break;case"month":u.ticks(Ge.every(h));break}}B.append("g").attr("class","grid").attr("transform","translate("+y+", "+T+")").call(u).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}l(G,"makeGrid");function H(y,T){let b=0;const x=Object.keys(V).map(k=>[k,V[k]]);B.append("g").selectAll("text").data(x).enter().append(function(k){const w=k[0].split(At.lineBreakRegex),c=-(w.length-1)/2,u=Y.createElementNS("http://www.w3.org/2000/svg","text");u.setAttribute("dy",c+"em");for(const[h,d]of w.entries()){const v=Y.createElementNS("http://www.w3.org/2000/svg","tspan");v.setAttribute("alignment-baseline","central"),v.setAttribute("x","10"),h>0&&v.setAttribute("dy","1em"),v.textContent=d,u.appendChild(v)}return u}).attr("x",10).attr("y",function(k,w){if(w>0)for(let c=0;c<w;c++)return b+=x[w-1][1],k[1]*y/2+b*y+T;else return k[1]*y/2+T}).attr("font-size",a.sectionFontSize).attr("class",function(k){for(const[w,c]of I.entries())if(k[0]===c)return"sectionTitle sectionTitle"+w%a.numberSectionStyles;return"sectionTitle"})}l(H,"vertLabels");function $(y,T,b,x){const k=i.db.getTodayMarker();if(k==="off")return;const w=B.append("g").attr("class","today"),c=new Date,u=w.append("line");u.attr("x1",S(c)+y).attr("x2",S(c)+y).attr("y1",a.titleTopMargin).attr("y2",x-a.titleTopMargin).attr("class","today"),k!==""&&u.attr("style",k.replace(/,/g,";"))}l($,"drawToday");function U(y){const T={},b=[];for(let x=0,k=y.length;x<k;++x)Object.prototype.hasOwnProperty.call(T,y[x])||(T[y[x]]=!0,b.push(y[x]));return b}l(U,"checkUnique")},"draw"),Fr={setConf:Ir,draw:Lr},Yr=l(e=>`
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

  .vert {
    stroke: ${e.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${e.vertLineColor} !important;
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
`,"getStyles"),Wr=Yr,Br={parser:Ut,db:Mr,renderer:Fr,styles:Wr};export{Br as diagram};

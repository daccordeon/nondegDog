def mysqrt(x): return np.sqrt((1.+0j)*x)

aux0=((1.+(np.exp(((0.+-2.j)*phiPump))))*((ga**2)*((W**2)*x)))+((0.+2.j)*((1.+(np.exp(((0.+-2.j)*phiPump))))*(ga*((W**3.)*x))));
aux1=(-2.*((mysqrt(2.))*((np.exp(((0.+-1.j)*phiPump)))*(rho*(ws**2)))))+aux0;
aux2=((1.+(np.exp(((0.+2.j)*phiPump))))*((ga**2)*((W**2)*x)))+((0.+-2.j)*((1.+(np.exp(((0.+2.j)*phiPump))))*(ga*((W**3.)*x))));
aux3=((-2.*((mysqrt(2.))*((np.exp(((0.+1.j)*phiPump)))*(rho*(ws**2)))))+aux2)-((1.+(np.exp(((0.+2.j)*phiPump))))*((W**4.)*x));
aux4=(-1.+Rpd)*((aux1-((1.+(np.exp(((0.+-2.j)*phiPump))))*((W**4.)*x)))*aux3);
aux5=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux6=((ga**2)*((((gbtot**2)+((0.+-2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+-1.j)*((gbtot**2)*W))+aux5)));
aux7=(ws**4.)+(((0.+2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux6));
aux8=(np.exp(((0.+1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux7))-((gbtot**2)*(W**2))));
aux9=((mysqrt(2.))*((np.exp(((0.+2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux8);
aux10=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+-1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux11=((ga**2)*((((gbtot**2)+((0.+2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+1.j)*((gbtot**2)*W))+aux10)));
aux12=(ws**4.)+(((0.+-2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux11));
aux13=(np.exp(((0.+-1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux12))-((gbtot**2)*(W**2))));
aux14=((mysqrt(2.))*((np.exp(((0.+-2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux13);
aux15=((-4.*(gbR*((gbtot-gbR)*aux4)))/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux9))/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux14);
aux16=(np.exp(((0.+-1.j)*phiPump)))*(((ga*(gbtot+((0.+1.j)*W)))+(((0.+1.j)*(gbtot*W))+(ws**2)))-(W**2));
aux17=((0.+1.j)*(ga*x))+((np.exp(((0.+-2.j)*phiPump)))*((((0.+-1.j)*ga)+W)*x));
aux18=(np.exp(((0.+1.j)*phiPump)))*(((ga*(gbtot+((0.+-1.j)*W)))+(((0.+-1.j)*(gbtot*W))+(ws**2)))-(W**2));
aux19=((0.+-1.j)*(ga*x))+((np.exp(((0.+2.j)*phiPump)))*((((0.+1.j)*ga)+W)*x));
aux20=(ga+((0.+1.j)*W))*((W**4.)*((((2.*aux16)+aux17)-(W*x))*(((2.*aux18)+aux19)-(W*x))));
aux21=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux22=((ga**2)*((((gbtot**2)+((0.+-2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+-1.j)*((gbtot**2)*W))+aux21)));
aux23=(ws**4.)+(((0.+2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux22));
aux24=(np.exp(((0.+1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux23))-((gbtot**2)*(W**2))));
aux25=((mysqrt(2.))*((np.exp(((0.+2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux24);
aux26=(-4.*(gbR*((gbtot-gbR)*((-1.+Rpd)*((ga+((0.+-1.j)*W))*aux20)))))/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux25);
aux27=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+-1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux28=((ga**2)*((((gbtot**2)+((0.+2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+1.j)*((gbtot**2)*W))+aux27)));
aux29=(ws**4.)+(((0.+-2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux28));
aux30=(np.exp(((0.+-1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux29))-((gbtot**2)*(W**2))));
aux31=((mysqrt(2.))*((np.exp(((0.+-2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux30);
aux32=(np.exp(((0.+-1.j)*phiPump)))*(((ga*(gbtot+((0.+1.j)*W)))+(((0.+1.j)*(gbtot*W))+(ws**2)))-(W**2));
aux33=((0.+1.j)*(ga*x))+((np.exp(((0.+-2.j)*phiPump)))*((((0.+-1.j)*ga)+W)*x));
aux34=(np.exp(((0.+1.j)*phiPump)))*(((ga*(gbtot+((0.+-1.j)*W)))+(((0.+-1.j)*(gbtot*W))+(ws**2)))-(W**2));
aux35=((0.+-1.j)*(ga*x))+((np.exp(((0.+2.j)*phiPump)))*((((0.+1.j)*ga)+W)*x));
aux36=(W**4.)*((ws**2)*((((2.*aux32)+aux33)-(W*x))*(((2.*aux34)+aux35)-(W*x))));
aux37=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux38=((ga**2)*((((gbtot**2)+((0.+-2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+-1.j)*((gbtot**2)*W))+aux37)));
aux39=(ws**4.)+(((0.+2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux38));
aux40=(np.exp(((0.+1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux39))-((gbtot**2)*(W**2))));
aux41=((mysqrt(2.))*((np.exp(((0.+2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux40);
aux42=(-4.*(ga*(gbR*((-1.+Rpd)*aux36))))/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux41);
aux43=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+-1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux44=((ga**2)*((((gbtot**2)+((0.+2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+1.j)*((gbtot**2)*W))+aux43)));
aux45=(ws**4.)+(((0.+-2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux44));
aux46=(np.exp(((0.+-1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux45))-((gbtot**2)*(W**2))));
aux47=((mysqrt(2.))*((np.exp(((0.+-2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux46);
aux48=(mysqrt(2.))*((np.exp(((0.+1.j)*phiPump)))*(rho*(gbtot+((0.+-1.j)*W))));
aux49=(np.exp(((0.+2.j)*phiPump)))*((((0.+1.j)*((mysqrt(2.))*rho))+((ga+((0.+-1.j)*W))*(W**2)))*x);
aux50=(2.*aux48)+(((((0.+-1.j)*((mysqrt(2.))*rho))+((ga+((0.+-1.j)*W))*(W**2)))*x)+aux49);
aux51=(mysqrt(2.))*((np.exp(((0.+-1.j)*phiPump)))*(rho*(gbtot+((0.+1.j)*W))));
aux52=(np.exp(((0.+-2.j)*phiPump)))*((((0.+-1.j)*((mysqrt(2.))*rho))+((ga+((0.+1.j)*W))*(W**2)))*x);
aux53=(2.*aux51)+(aux52+((((0.+1.j)*((mysqrt(2.))*rho))+((ga+((0.+1.j)*W))*(W**2)))*x));
aux54=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux55=((ga**2)*((((gbtot**2)+((0.+-2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+-1.j)*((gbtot**2)*W))+aux54)));
aux56=(ws**4.)+(((0.+2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux55));
aux57=(np.exp(((0.+1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux56))-((gbtot**2)*(W**2))));
aux58=((mysqrt(2.))*((np.exp(((0.+2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux57);
aux59=(-4.*(ga*(gbR*((-1.+Rpd)*((ws**2)*(aux50*aux53))))))/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux58);
aux60=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+-1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux61=((ga**2)*((((gbtot**2)+((0.+2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+1.j)*((gbtot**2)*W))+aux60)));
aux62=(ws**4.)+(((0.+-2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux61));
aux63=(np.exp(((0.+-1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux62))-((gbtot**2)*(W**2))));
aux64=((mysqrt(2.))*((np.exp(((0.+-2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux63);
aux65=((1.+(np.exp(((0.+-2.j)*phiPump))))*((ga**2)*((W**2)*x)))+((0.+2.j)*((1.+(np.exp(((0.+-2.j)*phiPump))))*(ga*((W**3.)*x))));
aux66=(-2.*((mysqrt(2.))*((np.exp(((0.+-1.j)*phiPump)))*(rho*(ws**2)))))+aux65;
aux67=((1.+(np.exp(((0.+2.j)*phiPump))))*((ga**2)*((W**2)*x)))+((0.+-2.j)*((1.+(np.exp(((0.+2.j)*phiPump))))*(ga*((W**3.)*x))));
aux68=(-2.*((mysqrt(2.))*((np.exp(((0.+1.j)*phiPump)))*(rho*(ws**2)))))+aux67;
aux69=(aux66-((1.+(np.exp(((0.+-2.j)*phiPump))))*((W**4.)*x)))*(aux68-((1.+(np.exp(((0.+2.j)*phiPump))))*((W**4.)*x)));
aux70=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux71=((ga**2)*((((gbtot**2)+((0.+-2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+-1.j)*((gbtot**2)*W))+aux70)));
aux72=(ws**4.)+(((0.+2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux71));
aux73=(np.exp(((0.+1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux72))-((gbtot**2)*(W**2))));
aux74=((mysqrt(2.))*((np.exp(((0.+2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux73);
aux75=(4.*((gbR**2)*((1.-Rpd)*aux69)))/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux74);
aux76=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+-1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux77=((ga**2)*((((gbtot**2)+((0.+2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+1.j)*((gbtot**2)*W))+aux76)));
aux78=(ws**4.)+(((0.+-2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux77));
aux79=(np.exp(((0.+-1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux78))-((gbtot**2)*(W**2))));
aux80=((mysqrt(2.))*((np.exp(((0.+-2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux79);
aux81=(-4.*(ga*(gbR*(W**3.))))+(((0.+-2.j)*(gbR*(W**4.)))+((mysqrt(2.))*(rho*(ws**2))));
aux82=(np.exp(((0.+-2.j)*phiPump)))*((((0.+2.j)*((ga**2)*(gbR*(W**2))))+aux81)*x);
aux83=(4.*(ga*(gbR*(W**3.))))+(((0.+2.j)*(gbR*(W**4.)))+((mysqrt(2.))*(rho*(ws**2))));
aux84=(-2.*(gbR*gbtot))+((gbtot**2)+(((0.+-2.j)*(gbR*W))+((0.+2.j)*(gbtot*W))));
aux85=((0.+2.j)*(gbR*(W**2)))+((W**3.)+(((0.+-1.j)*(gbR*(ws**2)))+(W*(x**2))));
aux86=(gbtot*(((0.+-2.j)*(gbR*W))+((-2.*(W**2))+(ws**2))))+((0.+-1.j)*(aux85-(W*(ws**2))));
aux87=((ga**2)*((aux84-(x**2))-(W**2)))+(2.*(ga*(((0.+1.j)*((gbtot**2)*W))+aux86)));
aux88=(2.*(gbtot*(W*((gbR*W)+(((0.+-1.j)*(W**2))+((0.+1.j)*(ws**2)))))))+(((W**2)*(x**2))+aux87);
aux89=((0.+-2.j)*(gbR*(W*(ws**2))))+((-2.*((W**2)*(ws**2)))+((ws**4.)+aux88));
aux90=(W**2)*((((0.+2.j)*(gbR*(W**3.)))+((W**4.)+aux89))-((gbtot**2)*(W**2)));
aux91=((((0.+-2.j)*((ga**2)*(gbR*(W**2))))+aux83)*x)+(2.*((np.exp(((0.+-1.j)*phiPump)))*aux90));
aux92=(4.*(ga*(gbR*(W**3.))))+(((0.+-2.j)*(gbR*(W**4.)))+((mysqrt(2.))*(rho*(ws**2))));
aux93=(-4.*(ga*(gbR*(W**3.))))+(((0.+2.j)*(gbR*(W**4.)))+((mysqrt(2.))*(rho*(ws**2))));
aux94=(np.exp(((0.+2.j)*phiPump)))*((((0.+-2.j)*((ga**2)*(gbR*(W**2))))+aux93)*x);
aux95=((gbtot**2)+((-2.*(gbtot*(gbR+((0.+1.j)*W))))+((0.+2.j)*(gbR*W))))-(x**2);
aux96=((0.+-2.j)*(gbR*(W**2)))+((W**3.)+(((0.+1.j)*(gbR*(ws**2)))+(W*(x**2))));
aux97=(gbtot*(((0.+2.j)*(gbR*W))+((-2.*(W**2))+(ws**2))))+((0.+1.j)*(aux96-(W*(ws**2))));
aux98=((W**2)*(x**2))+(((ga**2)*(aux95-(W**2)))+(2.*(ga*(((0.+-1.j)*((gbtot**2)*W))+aux97))));
aux99=(ws**4.)+((2.*(gbtot*(W*((gbR*W)+((0.+1.j)*((W**2)-(ws**2)))))))+aux98);
aux100=((0.+-2.j)*(gbR*(W**3.)))+((W**4.)+(((0.+2.j)*(gbR*(W*(ws**2))))+((-2.*((W**2)*(ws**2)))+aux99)));
aux101=2.*((np.exp(((0.+1.j)*phiPump)))*((W**2)*(aux100-((gbtot**2)*(W**2)))));
aux102=(aux82+aux91)*(((((0.+2.j)*((ga**2)*(gbR*(W**2))))+aux92)*x)+(aux94+aux101));
aux103=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux104=((ga**2)*((((gbtot**2)+((0.+-2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+-1.j)*((gbtot**2)*W))+aux103)));
aux105=(ws**4.)+(((0.+2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux104));
aux106=(np.exp(((0.+1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux105))-((gbtot**2)*(W**2))));
aux107=((mysqrt(2.))*((np.exp(((0.+2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux106);
aux108=(gbtot*((-2.*(W**2))+(ws**2)))+((0.+-1.j)*(W*(((W**2)+(x**2))-(ws**2))));
aux109=((ga**2)*((((gbtot**2)+((0.+2.j)*(gbtot*W)))-(x**2))-(W**2)))+(2.*(ga*(((0.+1.j)*((gbtot**2)*W))+aux108)));
aux110=(ws**4.)+(((0.+-2.j)*(gbtot*((W**3.)-(W*(ws**2)))))+(((W**2)*(x**2))+aux109));
aux111=(np.exp(((0.+-1.j)*phiPump)))*((W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+aux110))-((gbtot**2)*(W**2))));
aux112=((mysqrt(2.))*((np.exp(((0.+-2.j)*phiPump)))*(rho*((ws**2)*x))))+(2.*aux111);
aux113=(((1.-Rpd)*aux102)/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux107))/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux112);
aux114=(aux59/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux64))+(extSqzFactor*((aux75/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux80))+aux113));
aux115=(aux26/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux31))+((aux42/(((mysqrt(2.))*(rho*((ws**2)*x)))+aux47))+aux114);
output=mysqrt((Rpd+(aux15+aux115)));

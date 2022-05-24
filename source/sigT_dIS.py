def mysqrt(x): return np.sqrt((1.+0j)*x)

aux0=(2.*(ga*(gbtot*((-2.*(W**2))+(ws**2)))))+(((W**2)*(x**2))+((ga**2)*(((gbtot**2)-(x**2))-(W**2))));
aux1=(((W**4.)+((-2.*((W**2)*(ws**2)))+((ws**4.)+aux0)))-((gbtot**2)*(W**2)))*(np.cos(phiPump));
aux2=((gbtot*((W**2)-(ws**2)))+(ga*((((W**2)+(x**2))-(ws**2))-(gbtot**2))))-((ga**2)*gbtot);
aux3=((mysqrt(2.))*(rho*((ws**2)*(x*(np.cos((2.*phiPump)))))))+(-4.*((W**3.)*(aux2*(np.sin(phiPump)))));
aux4=((ga**2)*gbtot)+((gbtot*((ws**2)-(W**2)))+(ga*((((gbtot**2)+(ws**2))-(x**2))-(W**2))));
aux5=(2.*(ga*(gbtot*((-2.*(W**2))+(ws**2)))))+(((W**2)*(x**2))+((ga**2)*(((gbtot**2)-(x**2))-(W**2))));
aux6=(W**2)*(((W**4.)+((-2.*((W**2)*(ws**2)))+((ws**4.)+aux5)))-((gbtot**2)*(W**2)));
aux7=(aux6+((mysqrt(2.))*(rho*((ws**2)*(x*(np.cos(phiPump)))))))*(np.sin(phiPump));
aux8=(((((mysqrt(2.))*(rho*((ws**2)*x)))+((2.*((W**2)*aux1))+aux3))**2))+((((4.*((W**3.)*(aux4*(np.cos(phiPump)))))+(-2.*aux7))**2));
aux9=(np.cos(phiPump))*((((ga*gbtot)+(ws**2))-(ga*(x*(np.sin(phiPump)))))-(W**2));
aux10=(2.*(W*((np.sin(phiPump))*((ga+gbtot)-(x*(np.sin(phiPump)))))))+(2.*aux9);
aux11=(np.sin(phiPump))*((((W**2)+(ga*(x*(np.sin(phiPump)))))-(ws**2))-(ga*gbtot));
aux12=(2.*(W*((np.cos(phiPump))*((ga+gbtot)-(x*(np.sin(phiPump)))))))+(2.*aux11);
aux13=(Abs[B])*(((aux8/(gbR-(gbR*Rpd)))**-0.5)*(mysqrt(((aux10**2)+(aux12**2)))));
output=2.*((W**2)*(ws*aux13));

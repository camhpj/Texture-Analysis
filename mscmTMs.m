function [TMs,measout] = mscmTMs(MSCM)

MSCM = MSCM./sum(sum(MSCM)); %Normalize MSCM

size1 = size(MSCM,1);
size2 = size(MSCM,2);
size3 = size(MSCM,3);
size4 = size(MSCM,4);

mscmMean = mean(mean(MSCM)); % compute mean after norm
mscmVar = std(reshape(MSCM,size1*size2,size3)).^2;

px = squeeze(sum(MSCM,2));
py = squeeze(sum(MSCM,1));

pxplusy = zeros((size1*2 - 1),size3); %[1]
pxminusy = zeros((size1),size3); %[1]
for k = 1:size3
    for i = 1:size1
        for j = 1:size2
            px3(i,k) = px(i,k) + MSCM(i,j,k); 
            py3(i,k) = py(i,k) + MSCM(j,i,k); % taking i for j and j for i
            pxplusy((i+j)-1,k) = pxplusy((i+j)-1,k) + MSCM(i,j,k);
            pxminusy((abs(i-j))+1,k) = pxminusy((abs(i-j))+1,k) +...
                MSCM(i,j,k);
        end
    end
end

iMat  = repmat([1:size1]',1,size2);
jMat  = repmat([1:size2],size1,1);
iIdx   = jMat(:);
jIdx   = iMat(:);

xplusyIdx = [1:(2*(size1)-1)]';
xminusyIdx = [0:(size1-1)]';

multipleContr = abs(iMat - jMat).^2;
multipleDissi = abs(iMat - jMat);

ux = sum(sum(iMat.*MSCM)); 
uy = sum(sum(jMat.*MSCM)); 

sx = (sum(sum(((iMat - ux).^2).*MSCM))).^0.5;
sy = (sum(sum(((jMat - uy).^2).*MSCM))).^0.5;
    
corm = sum(sum(((iMat - ux).*(jMat - uy).*MSCM)));

%Measure Standatrd Haralick Features:
TMs.contr = sum(sum(multipleContr.*MSCM));
TMs.dissi = sum(sum(multipleDissi.*MSCM));
TMs.energ = sum(sum(MSCM.^2));
TMs.entro = - sum(sum((MSCM.*log(MSCM + eps))));
TMs.homom = sum(sum((MSCM./( 1 + multipleDissi))));
TMs.sosvh = sum(sum(MSCM.*((iMat - mscmMean).^2)));
TMs.indnc = sum(sum(MSCM./( 1 + (multipleDissi./size1) )));
TMs.idmnc = sum(sum(MSCM./( 1 + (multipleContr./(size2^2)))));
TMs.maxpr = max(max(MSCM));
TMs.autoc = sum(sum((iMat.*jMat.*MSCM)));
TMs.corrm = corm./(sx.*sy);
TMs.cprom = sum(sum(((iMat + jMat - ux - uy).^4).*MSCM)); 
TMs.cshad = sum(sum(((iMat + jMat - ux - uy).^3).*MSCM));        
TMs.savgh = sum((xplusyIdx + 1).*pxplusy);
   % the summation for savgh is for i from 2 to 2*Ng hence (i+1)
TMs.senth = -sum(pxplusy.*log(pxplusy + eps));
   % compute sum variance with the help of sum entropy
TMs.svarh = sum((((xplusyIdx+1)-TMs.senth).^2).*pxplusy);
TMs.denth = -sum((pxminusy).*log(pxminusy + eps));
TMs.dvarh = sum((xminusyIdx.^2).*pxminusy);

hXY = TMs.entro;
MSCMPrime  = permute(MSCM,[2,1,3]);
MSCMVec = MSCMPrime(:);

% pxIndexd = []; pyIndexd= [];
% for i = 1:size3
%     pxIndexd = cat(1,pxIndexd,px(iIdx));
%     pyIndexd = cat(1,pyIndexd,py(jIdx));
% end
%     
% hXY1 = -sum(reshape(MSCMVec.*log(pxIndexd.*pyIndexd + eps),[],size3));
% % hXY2 = -sum(reshape(px(iIdx),[],size3).*reshape(py(jIdx),[],size3).*...
% %     reshape(log(px(iIdx).*py(jIdx) + eps),[],size3));
% hXY2 = -sum(reshape(pxIndexd.*pyIndexd.*log(pxIndexd.*pyIndexd+eps),[],40));

for k = 1:size3    
    MSCMk  = MSCM(:,:,k)';
    MSCMv = MSCMk(:);
    hXY1(k) =  - sum(MSCMv.*log(px(iIdx,k).*py(jIdx,k) + eps));
    hXY2(k) =  - sum(px(iIdx,k).*py(jIdx,k).*...
        log(px(iIdx,k).*py(jIdx,k) + eps));
end

hX = -sum(px.*log(px + eps));
hY = -sum(py.*log(py + eps));   

TMs.inf1h = (reshape(hXY,1,[])-hXY1)./(max([hX;hY]));
TMs.inf2h = (1-exp(-2.*(hXY2-reshape(hXY,1,[])))).^0.5;


% measout = 1;
measout = [reshape(TMs.autoc,[],size3);... %autocorrelation
                reshape(TMs.contr,[],size3);%... %contrast (matlab)
                reshape(TMs.corrm,[],size3); ... %correlation (matlab)
                reshape(TMs.cprom,[],size3);... %cluster prominance
                reshape(TMs.cshad,[],size3);... %cluster shade
                reshape(TMs.dissi,[],size3);...
                reshape(TMs.energ,[],size3);...
                reshape(TMs.entro,[],size3);...
                reshape(TMs.homom,[],size3);... %Homogeneity
                reshape(TMs.maxpr,[],size3);... %max probability
                reshape(TMs.sosvh,[],size3);...
                reshape(TMs.savgh,[],size3);...
                reshape(TMs.senth,[],size3);...
                reshape(TMs.svarh,[],size3);...
                reshape(TMs.dvarh,[],size3);...
                reshape(TMs.denth,[],size3);...
                reshape(TMs.inf1h,[],size3);...
                reshape(TMs.inf2h,[],size3);...
                reshape(TMs.indnc,[],size3);...
                reshape(TMs.idmnc,[],size3)];
end



close all; clear all;

N=5;
as = linspace(.0125,.05,N);
fs = round(linspace(10,50,N));
t = linspace(0,2*pi,1000);
ps = @(a,f) ((.5 + a*sin(f*t)').*[cos(t)' sin(t)'])


for ii=1:numel(as)
    for jj=1:numel(fs)
        pij{ii,jj} = polyshape(ps(as(ii),fs(jj)));
        
        f1=figure; set(gcf,'color','w');
        axis equal; hold all; axis off; 
        plot(pij{ii,jj},'facecolor','k','facealpha',1); 
        exportgraphics(f1,sprintf('wavycircle_%d_%d.jpg',ii,jj))
        close(f1)
    end
end

p0 = polyshape(ps(0,1));        
f1=figure; set(gcf,'color','w');
axis equal; hold all; axis off; 
plot(p0,'facecolor','k','facealpha',1); 
exportgraphics(f1,sprintf('wavycircle_%d_%d.jpg',0,0))
close(f1)


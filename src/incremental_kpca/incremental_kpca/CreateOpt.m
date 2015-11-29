function [ OPTIONS ] = CreateOpt(T)
%OPTIONS Construct option structure for KPCA and IKPCA.

if T == 1
    R          = input('Value for R? ');
    KTYPE      = input('Value for KTYPE? ');
    KPARAM     = input('Value for KPARAM? ');
    DISP       = input('Value for DISP? ');
    OPTIONS = struct('R',R,'KTYPE',KTYPE,'KPARAM',KPARAM,'DISP',DISP);
elseif T == 2
    R          = input('Value for R? ');
    MAXLIB     = input('Value for MAXLIB? ');
    INC        = input('Value for INC? ');
    KTYPE      = input('Value for KTYPE? ');
    KPARAM     = input('Value for KPARAM? ');
    DISP       = input('Value for DISP? ');    
    OPTIONS = struct('R',R,'MAXLIB',MAXLIB,'INC',INC,'KTYPE',KTYPE,'KPARAM',KPARAM,'DISP',DISP);
end

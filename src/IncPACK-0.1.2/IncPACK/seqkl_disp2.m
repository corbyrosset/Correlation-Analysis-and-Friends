function seqkl_disp2(acc)
   fprintf(' Elapsed time: %f seconds.\n',acc.time);
   % values
   sstr = sprintf('  cond: %16.8e',acc.cond);
   if ~isnan(acc.dists),
      fprintf('%s      dists: %16.8e\n',sstr,acc.dists);
   else
      fprintf('%s\n',sstr);
   end
   % left subspace
   ustr = sprintf(' orthu: %16.8e   estorthu: %16.8e',...
                  acc.orthu,acc.estorthu);
   if ~isnan(acc.sdistu),
      fprintf('%s   distu: %16.8e\n',ustr,acc.sdistu);
   else
      fprintf('%s\n',ustr);
   end
   % right subspace
   vstr = sprintf(' orthv: %16.8e   estorthv: ________________',...
                  acc.orthv);
   if ~isnan(acc.sdistv),
      fprintf('%s   distv: %16.8e\n',vstr,acc.sdistv);
   else
      fprintf('%s\n',vstr);
   end

end   % end function prt_display2

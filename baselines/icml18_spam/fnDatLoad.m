% load the index number of data
% path: the file path of the data
function [datFeat, datLabel] = fnDatLoad(datPath, sIdx, eIdx, nDim)

%prepare the data.
nNum = eIdx - sIdx + 1;
datLabel = zeros(nNum, 1);

% some variables.
iIdx = [];
jIdx = [];
sVal = [];

nCnt = 1;
%open the file
fid = fopen(datPath);
while ~feof(fid)
    tLine = fgetl(fid);
    if ((nCnt >= sIdx) && (nCnt <= eIdx))
        if ~isempty(tLine)
            datLabel(nCnt - sIdx + 1, 1) = str2double(tLine(1,1:2));
            cBlank = find(tLine == ' ');
            if length(cBlank) > 1
                cDot = find(tLine == ':');
                %% if the last position does not contain a blank, add one.
                if length(cBlank) == length(cDot)
                    cBlank = [cBlank, length(tLine)];
                end
                % values between dots and blanks
                % get the iidx
                cTmp = [cBlank(1, 1:(length(cBlank)-1))', cDot(1,1:length(cDot))'];
                [nTs,~] = size(cTmp);
                for i = 1:nTs
                    iIdx = [iIdx, str2double(tLine(cTmp(i,1):(cTmp(i,2)-1)))];
                end
                
                % get the jidx
                jIdx = [jIdx, (nCnt-sIdx+1)*ones(1, nTs)];
                % get value
                cTmp = [cDot(1,:)', cBlank(1, 2:length(cBlank))'];
                [nTs,~] = size(cTmp);
                
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % normalize the data
                datItem = [];
                for i = 1:nTs
                    datItem = [datItem, str2double(tLine((cTmp(i,1)+1):cTmp(i,2)))];
                end
                
                datItem = datItem/norm(datItem, 2);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                
                sVal = [sVal, datItem];
            end        
        end
    end
    nCnt = nCnt + 1;
end

fclose(fid);

% sparse matrix
datFeat = sparse(jIdx, iIdx, sVal, nNum, nDim);


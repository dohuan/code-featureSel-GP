function y = icplxdual2D(w, J, Fsf, sf)



for j = 1:J
    for m = 1:3
        [w{j}{1}{1}{m} w{j}{2}{2}{m}] = pm(w{j}{1}{1}{m},w{j}{2}{2}{m});
        [w{j}{1}{2}{m} w{j}{2}{1}{m}] = pm(w{j}{1}{2}{m},w{j}{2}{1}{m});
    end
end

y = zeros(size(w{1}{1}{1}{1})*2);
for m = 1:2
    for n = 1:2
        lo = w{J+1}{m}{n};
        for j = J:-1:2
            lo = sfb2D(lo, w{j}{m}{n}, sf{m}, sf{n});
        end
        lo = sfb2D(lo, w{1}{m}{n}, Fsf{m}, Fsf{n});
        y = y + lo;
    end
end

% normalization
y = y/2;


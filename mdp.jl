module markov

export MDP, random_mdp, value_iteration

type MDP
  P
  R
  na
  ns
  gamma
end

MDP(P,R,gamma) = MDP(P,R,size(P,1),size(P,2),gamma)

function random_mdp(a,s)
    P = rand(a,s,s)
    R = rand(a,s,s)
    for i=1:a
      for j=1:s
        P[i,j,:] = P[i,j,:] / sum(P[i,j,:])
        R[i,j,:] = R[i,j,:] / sum(R[i,j,:])
      end
    end
    MDP(P,R,rand()) 
end

function imap(f,itr)
    @task for i=itr
        produce(f(i))
    end
end

function value_iteration(mdp::MDP, err)
    V = rand(mdp.ns)

    profits(s,a) = 
    @task for sn=1:mdp.ns
        produce(mdp.P[a,s,sn] * (mdp.R[a,s,sn] + mdp.gamma * V[sn]))
    end

    maxerr = err + 1
    while maxerr > err
      maxerr = 0.0
      for s=1:mdp.ns
        prev = V[s]
        V[s] = max(imap(sum, imap(a->profits(s,a), 1:mdp.na)))
        delta = abs(prev - V[s])
        maxerr = delta > maxerr ? delta : maxerr
      end
    end
    policy = [indmax([sum(profits(s,a)) for a=1:mdp.na]) for s=1:mdp.ns]
    return policy,V
end
  
end

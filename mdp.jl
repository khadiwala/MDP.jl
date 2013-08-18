module markov

export MDP, randommdp, valueiteration

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
    for i in 1:a
      for j in 1:s
        P[i,j,:] = P[i,j,:] / sum(P[i,j,:])
        R[i,j,:] = R[i,j,:] / sum(R[i,j,:])
      end
    end
    MDP(P,R,rand()) 
end

function value_iteration(mdp::MDP, err)
    V = rand(mdp.ns)
    profit(s1,s2,a) = mdp.P[a,s1,s2] * (mdp.R[a,s1,s2] + mdp.gamma * V[s2])
    error = err + 1
    count = 0
    while error > err
      count += 1
      error = 0.0
      for s=1:mdp.ns
        last = V[s]
        V[s] = max(sum(profit(s,s2,a) for s2 in 1:mdp.ns) for a in 1:mdp.na)
        error = abs(last - V[s]) > error ? abs(last - V[s]) : error
      end
    end
    policy = [indmax(sum(profit(s,s2,a) for s2 in 1:mdp.ns) for a in 1:mdp.na) for s in 1:mdp.ns]
    return V,policy
end
  
end

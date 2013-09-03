module markov

export MDP, random_mdp, value_iteration, policy_iteration

type MDP
  P
  R
  na
  ns
  gamma
end

MDP(P,R,gamma) = MDP(P,R,size(P,1),size(P,2),gamma)

function random_mdp(a,s)
    P,R = rand(a,s,s),rand(a,s,s)
    for i=1:a
      for j=1:s
        P[i,j,:] = P[i,j,:] / sum(P[i,j,:])
        R[i,j,:] = R[i,j,:] / sum(R[i,j,:])
      end
    end
    MDP(P,R,.95) 
end

function imap(f,itr)
    @task for i=itr
        produce(f(i))
    end
end

function argmax(itr,f)
    maxi = itr[1]
    maxf = f(itr[1])
    for i=itr[2:]
        val = f(i)
        if val > maxf
            maxi = i
            maxf = val
        end
    end
    return maxi
end

function value_iteration(mdp::MDP, err=.01)
    V = zeros(mdp.ns)

    bellman(s,a) = 
    @task for sn=1:mdp.ns
        produce(mdp.P[a,s,sn] * (mdp.R[a,s,sn] + mdp.gamma * V[sn]))
    end
    
    # the maximum delta b/w iterations to ensure error < err
    errdel = err * (1 - mdp.gamma) / (2 * mdp.gamma)

    maxerr = errdel + 1
    while maxerr > errdel
      maxerr = 0.0
      for s=1:mdp.ns
        prev = V[s]
        V[s] = max(imap(sum, imap(a->bellman(s,a), 1:mdp.na)))
        delta = abs(prev - V[s])
        maxerr = delta > maxerr ? delta : maxerr
      end
    end
    policy = [argmax(1:mdp.na,a->sum(bellman(s,a))) for s=1:mdp.ns]
    return policy,V
end

function policy_iteration(mdp::MDP)
    policy = ones(mdp.ns)
    V = zeros(mdp.ns)

    bellman(s,a) = 
    @task for sn=1:mdp.ns
        produce(mdp.P[a,s,sn] * (mdp.R[a,s,sn] + mdp.gamma * V[sn]))
    end

    done = false
    while !done
      V = [sum(imap(x->mdp.P[policy[s],s,x]*(mdp.R[policy[s],s,x] + mdp.gamma*V[x]),1:mdp.ns)) for s=1:mdp.ns]
      done = true
      for s=1:mdp.ns
        prev = policy[s]
        policy = [argmax(1:mdp.na,a->sum(bellman(s,a))) for s=1:mdp.ns]
        done &= policy[s] == prev
      end
    end
    return policy, V
end

function q_learner(s,a,alpha,gamma)
  Q = zeros(s,a)
  q_learn(state,f) =
    choice = argmax(1:a,x->Q[state,x])
    result,reward = f(state,choice)
    Q[s,a] += alpha*(reward + gamma*(argmax(1:a,x->Q[result,x]) - Q[state,choice]))
    return result
  end
  return q_learn
end

  
end

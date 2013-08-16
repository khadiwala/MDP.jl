module markov

export MDP, randommdp

type MDP
  P
  R
  gamma
end

function randommdp(a,s)
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

function valueiteration(mdp::MDP)
  
end

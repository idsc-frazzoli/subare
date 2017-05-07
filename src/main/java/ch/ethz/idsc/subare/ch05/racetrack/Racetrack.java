// code by jph
package ch.ethz.idsc.subare.ch05.racetrack;

import java.util.List;
import java.util.Random;

import ch.ethz.idsc.subare.core.EpisodeInterface;
import ch.ethz.idsc.subare.core.EpisodeSupplier;
import ch.ethz.idsc.subare.core.MonteCarloInterface;
import ch.ethz.idsc.subare.core.PolicyInterface;
import ch.ethz.idsc.subare.core.StandardModel;
import ch.ethz.idsc.subare.core.mc.MonteCarloEpisode;
import ch.ethz.idsc.subare.core.util.StateActionMap;
import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.subare.util.Index;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.ZeroScalar;
import ch.ethz.idsc.tensor.alg.Array;
import ch.ethz.idsc.tensor.alg.Dimensions;
import ch.ethz.idsc.tensor.alg.Join;
import ch.ethz.idsc.tensor.sca.Clip;
import ch.ethz.idsc.tensor.sca.Decrement;

/** Exercise 5.8: Racetrack (programming), p. 119
 * 
 * Figure 5.6
 * 
 * the book states that the velocity components should be non-negative
 * the track layout however encourages nudging in the negative direction
 * so we make a compromise by using the following integration procedure
 * p' = p + v + a
 * v' = clip(v + a) */
class Racetrack implements StandardModel, MonteCarloInterface, EpisodeSupplier {
  public static final Tensor WHITE = Tensors.vector(255, 255, 255, 255);
  public static final Tensor RED = Tensors.vector(255, 0, 0, 255);
  public static final Tensor GREEN = Tensors.vector(0, 255, 0, 255);
  public static final Tensor BLACK = Tensors.vector(0, 0, 0, 255);
  public static final Scalar MINUS_ONE = RealScalar.ONE.negate();
  // ---
  final Clip clipPositionY;
  final Clip clipSpeed;
  final Tensor dimensions;
  private final Tensor states = Tensors.empty(); // (px, py, vx, vy)
  final Tensor statesStart = Tensors.empty();
  final Tensor statesTerminal = Tensors.empty();
  private final Tensor actions = Tensor.of(Array.of(Tensors::vector, 3, 3).flatten(1)).map(Decrement.ONE).unmodifiable();
  final Index statesIndex;
  final Index statesStartIndex;
  final Index statesTerminalIndex;
  Random random = new Random();
  private final StateActionMap stateActionMap;

  public Racetrack(Tensor track, int maxSpeed) {
    List<Integer> list = Dimensions.of(track);
    dimensions = Tensors.vector(list.subList(0, 2)).map(Decrement.ONE);
    clipPositionY = Clip.function(ZeroScalar.get(), dimensions.Get(1));
    clipSpeed = Clip.function(0, maxSpeed);
    for (int x = 0; x < list.get(0); ++x)
      for (int y = 0; y < list.get(1); ++y) {
        final Tensor rgba = track.get(x, y).unmodifiable();
        if (!rgba.equals(WHITE)) {
          final Tensor pstate = Tensors.vector(x, y);
          if (rgba.equals(BLACK))
            for (int vx = 0; vx <= maxSpeed; ++vx)
              for (int vy = 0; vy <= maxSpeed; ++vy)
                if (vx != 0 || vy != 0)
                  states.append(Join.of(pstate, Tensors.vector(vx, vy)));
          // ---
          if (rgba.equals(GREEN)) {
            Tensor state = Join.of(pstate, Tensors.vector(0, 0));
            states.append(state);
            statesStart.append(state);
          }
          // ---
          if (rgba.equals(RED))
            for (int vx = 0; vx <= maxSpeed; ++vx)
              for (int vy = 0; vy <= maxSpeed; ++vy)
                if (vx != 0 || vy != 0) {
                  Tensor state = Join.of(pstate, Tensors.vector(vx, vy));
                  states.append(state);
                  statesTerminal.append(state);
                }
        }
      }
    statesIndex = Index.build(states);
    statesStartIndex = Index.build(statesStart);
    statesTerminalIndex = Index.build(statesTerminal);
    GlobalAssert.of(Dimensions.isArray(states));
    GlobalAssert.of(Dimensions.isArray(actions));
    stateActionMap = StateActionMap.build(this, actions, this);
  }

  @Override
  public Tensor states() {
    return states.unmodifiable();
  }

  @Override
  public Tensor actions(Tensor state) {
    return stateActionMap.actions(state);
  }

  @Override
  public Scalar qsa(Tensor state, Tensor action, Tensor gvalues) {
    // general term in bellman equation:
    // Sum_{s',r} p(s',r | s,a) * (r + gamma * v_pi(s'))
    // simplifies here to
    // 1 * (r + gamma * v_pi(s'))
    Tensor next = move(state, action);
    int nextI = statesIndex.of(next);
    return reward(state, action, next).add(gvalues.get(nextI));
  }

  private static Tensor shift(Tensor state, Tensor action) {
    Tensor pos = state.extract(0, 2);
    Tensor vel = state.extract(2, 4);
    vel = vel.add(action);
    return Join.of(pos.add(vel), vel);
  }

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    Tensor next = shift(state, action); // add velocity
    next.set(clipPositionY, 1);
    next.set(clipSpeed, 2); // vx
    next.set(clipSpeed, 3); // vy
    if (statesIndex.containsKey(next))
      // TODO collision checking
      return next;
    return statesStart.get(random.nextInt(statesStart.length()));
  }

  @Override
  public Scalar reward(Tensor state, Tensor action, Tensor next) {
    if (!isTerminal(state) && isTerminal(next))
      return RealScalar.ONE;
    return isTerminal(next) ? ZeroScalar.get() : MINUS_ONE;
  }

  boolean isStart(Tensor state) {
    return statesStartIndex.containsKey(state);
  }

  @Override
  public boolean isTerminal(Tensor state) {
    return statesTerminalIndex.containsKey(state);
  }

  @Override
  public EpisodeInterface kickoff(PolicyInterface policyInterface) {
    Tensor start = statesStart.get(random.nextInt(statesStart.length()));
    if (isTerminal(start))
      throw new RuntimeException();
    return new MonteCarloEpisode(this, policyInterface, start);
  }
}

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

class Racetrack implements StandardModel, MonteCarloInterface, EpisodeSupplier {
  public static final Tensor WHITE = Tensors.vector(255, 255, 255, 255);
  public static final Tensor RED = Tensors.vector(255, 0, 0, 255);
  public static final Tensor GREEN = Tensors.vector(0, 255, 0, 255);
  public static final Tensor BLACK = Tensors.vector(0, 0, 0, 255);
  public static final Scalar MINUS_ONE = RealScalar.ONE.negate();
  // ---
  final int MAX_SPEED;
  final Clip clip;
  final Tensor dimensions;
  final Tensor states = Tensors.empty(); // (px, py, vx, vy)
  final Tensor statesStart = Tensors.empty();
  final Tensor statesTerminal = Tensors.empty();
  final Tensor actions = Tensor.of(Array.of(Tensors::vector, 3, 3).flatten(1)).map(Decrement.ONE);
  final Index statesIndex;
  final Index statesStartIndex;
  final Index statesTerminalIndex;
  Random random = new Random();

  public Racetrack(Tensor track, int MAX_SPEED) {
    this.MAX_SPEED = MAX_SPEED;
    clip = Clip.function(0, MAX_SPEED);
    List<Integer> list = Dimensions.of(track);
    dimensions = Tensors.vector(list.subList(0, 2)).map(Decrement.ONE);
    System.out.println(dimensions);
    for (int x = 0; x < list.get(0); ++x)
      for (int y = 0; y < list.get(1); ++y) {
        final Tensor rgba = track.get(x, y).unmodifiable();
        if (!rgba.equals(WHITE)) {
          final Tensor pstate = Tensors.vector(x, y);
          if (rgba.equals(BLACK))
            for (int vx = 0; vx < MAX_SPEED; ++vx)
              for (int vy = 0; vy < MAX_SPEED; ++vy)
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
            for (int vx = 0; vx < MAX_SPEED; ++vx)
              for (int vy = 0; vy < MAX_SPEED; ++vy)
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
    System.out.println(states.length());
    System.out.println(statesStart);
    // System.out.println(statesTerminal);
    // System.out.println(actions.length());
    System.out.println(Dimensions.isArray(states));
    System.out.println(Dimensions.isArray(actions));
  }

  @Override
  public Tensor states() {
    return states;
  }

  @Override
  public Tensor actions(Tensor state) {
    if (isTerminal(state))
      return Tensors.of(Tensors.vector(0, 0)); // no gas in finish zone
    Tensor filter = Tensors.empty();
    for (Tensor action : actions) {
      Tensor next = state.add(Join.of(Array.zeros(2), action)); // add velocity
      Tensor effective = move(state, action);
      if (next.equals(effective))
        filter.append(action);
    }
    if (filter.length() == 0)
      throw new RuntimeException("no actions for " + state);
    return filter;
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

  @Override
  public Tensor move(Tensor state, Tensor action) {
    if (isTerminal(state))
      return state;
    Tensor delta = Join.of(Array.zeros(2), action);
    // System.out.println(state + " " + action + " " + delta);
    Tensor next = state.add(delta); // add velocity
    next.set(clip, 2); // vx
    next.set(clip, 3); // vy
    // TODO collision checking
    if (statesIndex.containsKey(next))
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

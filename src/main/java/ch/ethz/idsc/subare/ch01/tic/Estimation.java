// code by jph
package ch.ethz.idsc.subare.ch01.tic;

import java.io.File;
import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/** A value function specifies what is good in the long run.
 * Roughly speaking, the value of a state is the total amount of reward
 * an agent can expect to accumulate over the future, starting from that state. */
/* package */ class Estimation implements Serializable {
  private final Map<State, Double> map = new HashMap<>();

  /** Assuming we always play Xs, then for
   * all states with three Xs in a row the probability of winning is 1,
   * because we have already won.
   * Similarly, for all states with three Os in a row, or that are “filled up,”
   * the correct probability is 0, as we cannot win from them.
   * We set the initial values of all the other states to 0.5,
   * representing a guess that we have a 50% chance of winning. */
  public void init(int symbol) {
    for (State state : AllStates.INSTANCE.getEquivalenceSet()) {
      if (Objects.isNull(state.winner)) {
        map.put(state, 0.5);
      } else
        switch (state.winner) {
        case 0:
          map.put(state, 0.0); // not sure how to rate a draw
          break;
        default:
          map.put(state, state.winner == symbol ? 1.0 : 0.0);
          break;
        }
    }
  }

  private static State normalize(State state) {
    // return AllStates.instance.getEquivalent(state);
    return AllStates.INSTANCE.getRepresentative(state);
  }

  public double get(State state) {
    state = normalize(state);
    return map.containsKey(state) //
        ? map.get(state)
        : 0.0;
  }

  public void put(State state, double value) {
    state = normalize(state);
    map.put(state, value);
  }

  static File of(int symbol) {
    return new File("policy" + symbol + ".bin");
  }

  public static Estimation load(int symbol) {
    try {
      // Export.object(file, object);(object)
      // return (Estimation) ObjectExchange.readObject(of(symbol));
      return null;
    } catch (Exception myException) {
      myException.printStackTrace();
    }
    return new Estimation();
  }

  public static void save(Estimation estimation, int symbol) {
    // ObjectExchange.writeObject(of(symbol), estimation);
  }
}

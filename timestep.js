// Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.

pi = 3.141592653589793
boundary = [100, 100, 100] // (x, y, z), y pointing up
cell_size = 2.51
cell_recpr = 1.0 / cell_size

Array.prototype.norm = function () {
    var s = 0;
    for (var i = 0; i < this.length; i++) {
        s += this[i] * this[i];
    }
    return Math.sqrt(s);
}

function round_up(f, s) {
    return (Math.floor(f * cell_recpr / s) + 1) * s
}

grid_size = [round_up(boundary[0], 1), round_up(boundary[1], 1), round_up(boundary[2], 1)]
console.log(grid_size)

dim = 3
num_particles_x = 10
num_particles_y = 10
num_particles_z = 10
num_particles = num_particles_x * num_particles_y * num_particles_y
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 100.0
epsilon = 1e-5
particle_radius_in_world = 0.3

// PBF params
h = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 5
corr_deltaQ_coeff = 0.3
corrK = 0.001
// Need ti.pow()
// corrN = 4.0
neighbor_radius = h * 1.05

poly6_factor = 315.0 / 64.0 / pi
spiky_grad_factor = -45.0 / pi

var old_positions
var positions
var velocities
var lambdas
var position_deltas
// 0: x-pos, 1: timestep in sin()
var board_states


var grid_num_particles
var grid2particles
var particle_num_neighbors
var particle_neighbors

function norm(a, b, dim) {
    var s = 0;
    for (var i = 0; i < dim; i++) {
        s += Math.pow(a[i] - b[i], 2)
    }
    s = Math.sqrt(s);
    return s;
}

function createVariables() {
    positions = new Array(num_particles)
    old_positions = new Array(num_particles)
    velocities = new Array(num_particles)
    position_deltas = new Array(num_particles)
    for (var i = 0; i < num_particles; i++) {
        positions[i] = new Array(dim);
        old_positions[i] = new Array(dim);
        velocities[i] = new Array(dim);
        position_deltas[i] = new Array(dim);
    }

    lambdas = new Array(num_particles)
    board_states = new Array(dim);

    grid_num_particles = new Array(grid_size[0]) // x
    for (var i = 0; i < grid_size[0]; i++) {
        grid_num_particles[i] = new Array(grid_size[1]) // y
        for (var j = 0; j < grid_size[1]; j++) {
            grid_num_particles[i][j] = new Array(grid_size[2]) // z;
        }
    }

    grid2particles = new Array(grid_size[0]) // x
    for (var i = 0; i < grid_size[0]; i++) {
        grid2particles[i] = new Array(grid_size[1]) // y
        for (var j = 0; j < grid_size[1]; j++) {
            grid2particles[i][j] = new Array(grid_size[2]) // z;
            for (var k = 0; k < grid_size[2]; k++) {
                grid2particles[i][j][k] = new Array(max_num_particles_per_cell) // hash
            }
        }
    }

    particle_num_neighbors = new Array(num_particles)
    particle_neighbors = new Array(num_particles)
    for (var i = 0; i < num_particles; i++) {
        particle_neighbors[i] = new Array(max_num_neighbors);
    }

}


function poly6_value(s, h) {
    result = 0.0
    if (0 < s && s < h) {
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    }
    return result
}


function spiky_gradient(r, h) {
    result = [0.0, 0.0, 0.0]
    r_len = r.norm()
    if (0 < r_len && r_len < h) {
        var x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result[0] = r[0] * g_factor / r_len
        result[1] = r[1] * g_factor / r_len
        result[2] = r[2] * g_factor / r_len
    }
    return result
}

function compute_scorr(pos_ji) {
    // Eq (13)
    x = poly6_value(pos_ji.norm(), h) / poly6_value(corr_deltaQ_coeff * h, h)
    // pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x
}

function get_cell(pos) {
    return [Math.floor(pos[0] * cell_recpr), Math.floor(pos[1] * cell_recpr), Math.floor(pos[2] * cell_recpr)]
}


function is_in_grid(c) {
    // @c: Vector(i32)
    return 0 <= c[0] && c[0] < grid_size[0] && 0 <= c[1] && c[1] < grid_size[1] && 0 <= c[2] && c[2] < grid_size[2]
}

function confine_position_to_boundary(p) {
    bmin = particle_radius_in_world
    bmax = boundary[0] - particle_radius_in_world
    //
    ret = new Array(3)
    for (var i = 0; i < dim; i++) {
        // Use randomness to prevent particles from sticking into each other after clamping
        if (p[i] <= bmin)
            ret[i] = bmin
        else if (p[i] >= bmax)
            ret[i] = bmax
        else
            ret[i] = p[i]
    }
    return ret
}

function move_board() {
    // probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 8.0
    if (b[1] >= 2 * period)
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b
}

function prologue() {
    // save old positions
    for (var p_i = 0; p_i < num_particles; p_i++) {
        for (var j = 0; j < dim; j++) {
            old_positions[p_i][j] = positions[p_i][j];
        }
    }
    // apply gravity within boundary
    g = [0, -9.8, 0]
    for (var p_i = 0; p_i < num_particles; p_i++) {
        pos = new Array(3)
        pos[0] = positions[p_i][0]
        pos[1] = positions[p_i][1]
        pos[2] = positions[p_i][2]
        vel = new Array(3)
        vel[0] = velocities[p_i][0]
        vel[1] = velocities[p_i][1]
        vel[2] = velocities[p_i][2]
        for (var j = 0; j < dim; j++) {
            vel[j] += g[j] * time_delta
            pos[j] += vel[j] * time_delta
        }
        cptb = confine_position_to_boundary(pos)
        positions[p_i][0] = cptb[0]
        positions[p_i][1] = cptb[1]
        positions[p_i][2] = cptb[2]
    }

    // clear neighbor lookup table
    for (var i = 0; i < grid_size[0]; i++)
        for (var j = 0; j < grid_size[1]; j++)
            for (var k = 0; k < grid_size[2]; k++)
                grid_num_particles[i][j][k] = 0 // z;

    for (var p_i = 0; p_i < num_particles; p_i++)
        for (var j = 0; j < num_particles; j++)
            particle_neighbors[p_i][j] = -1;

    // update grid
    for (var p_i = 0; p_i < num_particles; p_i++) {
        c = new Array(3)
        c = get_cell(positions[p_i])
        if(!is_in_grid(c)) {
            console.log(c + "; " + p_i)
            
            console.log(confine_position_to_boundary(positions[p_i]))
        }

        offs = grid_num_particles[c[0]][c[1]][c[2]]
        if(offs >= max_num_particles_per_cell - 1) continue;
        grid_num_particles[c[0]][c[1]][c[2]] += 1;
        grid2particles[c[0]][c[1]][c[2]][offs] = p_i;
    }


    for (var p_i = 0; p_i < num_particles; p_i++) {
        pos_i = new Array(3)
        pos_i[0] = positions[p_i][0]
        pos_i[1] = positions[p_i][1]
        pos_i[2] = positions[p_i][2]
        cell = get_cell(pos_i)
        nb_i = 0
        for (var offsX = -1; offsX < 2; offsX++) {
            for (var offsY = -1; offsY < 2; offsY++) {
                for (var offsZ = -1; offsZ < 2; offsZ++) {
                    cell_to_check = [offsX + cell[0], offsY + cell[1], offsZ + cell[2]]
                    if (is_in_grid(cell_to_check)) {
                        for (var j = 0; j < grid_num_particles[cell_to_check[0]][cell_to_check[1]][cell_to_check[2]]; j++) {
                            p_j = grid2particles[cell_to_check[0]][cell_to_check[1]][cell_to_check[2]][j];
                            if (nb_i < max_num_neighbors && p_j != p_i &&
                                norm(pos_i, positions[p_j], dim) < neighbor_radius) {
                                particle_neighbors[p_i][nb_i] = p_j
                                nb_i += 1
                            }
                        }
                    }
                }
            }
        }
        particle_num_neighbors[p_i] = nb_i
    }
    // find particle neighbors
}

function substep() {
    for (var p_i = 0; p_i < num_particles; p_i++) {
        pos_i = positions[p_i]
        grad_i = [0.0, 0.0, 0.0]
        sum_gradient_sqr = 0.0
        density_constraint = 0.0
        for (var j = 0; j < particle_num_neighbors[p_i]; j++) {
            p_j = particle_neighbors[p_i][j]
            pos_ji = [pos_i[0] - positions[p_j][0], pos_i[1] - positions[p_j][1], pos_i[2] - positions[p_j][2]]
            grad_j = spiky_gradient(pos_ji, h)
            grad_i[0] += grad_j[0]
            grad_i[1] += grad_j[1]
            grad_i[2] += grad_j[2]
            sum_gradient_sqr += Math.pow(grad_j.norm(), 2)
            // Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h)
        }

        // Eq(1)
        density_constraint = (mass * density_constraint / rho0) - 1.0

        sum_gradient_sqr += Math.pow(grad_i.norm(), 2)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr +
            lambda_epsilon)
    }
    // compute position deltas
    // Eq(12), (14)
    for (var p_i = 0; p_i < num_particles; p_i++) {
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = [0.0, 0.0, 0.0]
        for (var j = 0; j < particle_num_neighbors[p_i]; j++) {
            p_j = particle_neighbors[p_i][j]
            if (p_j < 0)
                break;
            lambda_j = lambdas[p_j]
            pos_ji = new Array(3)
            pos_ji[0] = pos_i[0] - positions[p_j][0]
            pos_ji[1] = pos_i[1] - positions[p_j][1]
            pos_ji[2] = pos_i[2] - positions[p_j][2]
            scorr_ij = compute_scorr(pos_ji)
            grad = spiky_gradient(pos_ji, h);
            pos_delta_i[0] += (lambda_i + lambda_j + scorr_ij) * grad[0]
            pos_delta_i[1] += (lambda_i + lambda_j + scorr_ij) * grad[1]
            pos_delta_i[2] += (lambda_i + lambda_j + scorr_ij) * grad[2]
        }

        pos_delta_i[0] /= rho0
        pos_delta_i[1] /= rho0
        pos_delta_i[2] /= rho0
        position_deltas[p_i][0] = pos_delta_i[0]
        position_deltas[p_i][1] = pos_delta_i[1]
        position_deltas[p_i][2] = pos_delta_i[2]
    }
    // apply position deltas
    for (var i = 0; i < num_particles; i++){
        positions[i][0] += position_deltas[i][0]
        positions[i][1] += position_deltas[i][1]
        positions[i][2] += position_deltas[i][2]
    }
}

function epilogue() {
    // confine to boundary
    for (var i = 0; i < num_particles; i++) {
        cptb = confine_position_to_boundary(positions[i])
        positions[i][0] = cptb[0]
        positions[i][1] = cptb[1]
        positions[i][2] = cptb[2]
    }
    // update velocities
    for (var i = 0; i < num_particles; i++) {
        velocities[i][0] = (positions[i][0] - old_positions[i][0]) / time_delta
        velocities[i][1] = (positions[i][1] - old_positions[i][1]) / time_delta
        velocities[i][2] = (positions[i][2] - old_positions[i][2]) / time_delta
    }
    // no vorticity/xsph because we cannot do cross product in 2D...
}


function run_pbf() {
    prologue()
    for (var i = 0; i < pbf_num_iters; i++) {
        substep()
    }
    epilogue()
}


function init_particles() {
    for (var i = 0; i < num_particles; i++) {
        delta = h * 0.8
        offs = [(boundary[0] - delta * num_particles_x) * 0.5, boundary[1] * 0.5, boundary[2] * 0.5]
        var x = Math.floor(i / num_particles_y) % num_particles_x;
        var y = (i % num_particles_y);
        var z = Math.floor(Math.floor(i / num_particles_x) / num_particles_y)
        positions[i][0] = x * delta + offs[0]
        positions[i][1] = y * delta + offs[1]
        positions[i][2] = z * delta + offs[2]
        for (var c = 0; c < dim; c++)
            velocities[i][c] = 0
    }
    //board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])
}


function print_stats() {
    console.log('PBF stats:')
    max = 0;
    avg = 0;
    for (var i = 0; i < grid_size[0]; i++)
        for (var j = 0; j < grid_size[1]; j++)
            for (var k = 0; k < grid_size[2]; k++) {
                if (grid_num_particles[i][j][k] > max)
                    max = grid_num_particles[i][j][k];
                avg += grid_num_particles[i][j][k];
            }
    avg /= (grid_size[0] * grid_size[1] * grid_size[2])
    console.log("  #particles per cell: avg={" + avg + "} max={" + max + "}")
    max = 0;
    avg = 0;
    for (var i = 0; i < num_particles; i++) {
        avg += particle_num_neighbors[i];
        if (particle_num_neighbors[i] > max)
            max = particle_num_neighbors[i]
    }
    avg
    console.log("  #neighbors per particle: sum={" + avg + "} max={" + max + "}")
    console.log(positions[0])
}


function main() {
    createVariables()
    init_particles()
    frame = 0;
    while (true) {
        frame++;
        //move_board()
        run_pbf()
        if (frame % 20 == 1)
            print_stats()
        if(frame > 1000)
        break;
    }
}

main()
